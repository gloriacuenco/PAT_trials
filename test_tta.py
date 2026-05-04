"""
Test-Time Augmentation (TTA) inference script.

Strategy:
  For each TTA scale in cfg.TEST.TTA_SCALES:
    - Build a dataloader with that scale's transform
    - Extract features for all query + gallery images
  Average all per-scale feature tensors → compute distance matrix.

TTA scales default to [224, 256]:
  - 224: standard resize → direct inference
  - 256: resize to 256 first, then center-crop to 224 → slightly zoomed-out view

We intentionally avoid horizontal flip TTA because traffic signs are
orientation-sensitive and make up ~74% of the dataset.

Usage:
  python test_tta.py --config_file ./config/UrbanElementsReID_test.yml
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from data.common import CommDataset
from data.transforms import build_tta_transforms
from data.build_DG_dataloader import fast_batch_collator
from data.samplers import InferenceSampler
from model import make_model
from utils.logger import setup_logger
from utils.metrics import euclidean_distance, eval_func
from utils.reranking import re_ranking


# ---------------------------------------------------------------------------- #
# TTA-aware dataloader builder
# ---------------------------------------------------------------------------- #

def build_tta_test_loader(cfg, dataset_name, transform):
    """
    Build a test dataloader with a custom transform (for TTA).
    Mirrors build_reid_test_loader but accepts an explicit transform.
    """
    from data.datasets import DATASET_REGISTRY

    _root = cfg.DATASETS.ROOT_DIR
    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    test_items = dataset.query + dataset.gallery
    num_query = len(dataset.query)

    test_set = CommDataset(test_items, transform=transform, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=fast_batch_collator,
    )
    return test_loader, num_query


# ---------------------------------------------------------------------------- #
# Feature extraction helper
# ---------------------------------------------------------------------------- #

def extract_features(model, loader, device, feat_norm):
    """Run one pass of inference and return normalized feature tensor."""
    model.eval()
    feats, pids, camids = [], [], []

    with torch.no_grad():
        for informations in loader:
            img = informations['images'].to(device)
            pid = informations['targets']
            camid = informations['camid']

            feat = model(img)
            # model may return tuple during eval — take last element
            if isinstance(feat, (tuple, list)):
                feat = feat[-1]

            feats.append(feat.cpu())
            pids.extend(np.asarray(pid))
            camids.extend(np.asarray(camid))

    feats = torch.cat(feats, dim=0)
    if feat_norm:
        feats = F.normalize(feats, dim=1, p=2)
    return feats, np.asarray(pids), np.asarray(camids)


# ---------------------------------------------------------------------------- #
# TTA Inference
# ---------------------------------------------------------------------------- #

def do_inference_tta(cfg, model, dataset_name):
    device = "cuda"
    logger = setup_logger("PAT.test_tta", cfg.LOG_ROOT, if_train=False)
    logger.info("=" * 60)
    logger.info("TTA Inference — scales: {}".format(cfg.TEST.TTA_SCALES))
    logger.info("=" * 60)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    tta_transforms = build_tta_transforms(cfg)
    logger.info("Built {} TTA transforms".format(len(tta_transforms)))

    # Collect features for each TTA scale
    all_feats = []
    pids_ref, camids_ref, num_query_ref = None, None, None

    t0 = time.time()
    for i, tfm in enumerate(tta_transforms):
        scale = cfg.TEST.TTA_SCALES[i]
        logger.info("Running scale {} ({}/{})...".format(scale, i + 1, len(tta_transforms)))

        loader, num_query = build_tta_test_loader(cfg, dataset_name, tfm)

        feats, pids, camids = extract_features(
            model, loader, device, feat_norm=cfg.TEST.FEAT_NORM
        )

        if num_query_ref is None:
            num_query_ref = num_query
            pids_ref = pids
            camids_ref = camids
        else:
            # Sanity check: all scales must produce same number of samples
            assert len(feats) == len(all_feats[0]), (
                f"Feature count mismatch at scale {scale}: "
                f"expected {len(all_feats[0])}, got {len(feats)}"
            )

        all_feats.append(feats)
        logger.info("  → Features shape: {}".format(feats.shape))

    # Average features across all TTA scales
    feats_avg = torch.stack(all_feats, dim=0).mean(dim=0)
    if cfg.TEST.FEAT_NORM:
        feats_avg = F.normalize(feats_avg, dim=1, p=2)

    logger.info("Averaged features from {} scales. Shape: {}".format(
        len(all_feats), feats_avg.shape))

    # Split into query / gallery
    qf = feats_avg[:num_query_ref]
    gf = feats_avg[num_query_ref:]
    q_pids = pids_ref[:num_query_ref]
    g_pids = pids_ref[num_query_ref:]
    q_camids = camids_ref[:num_query_ref]
    g_camids = camids_ref[num_query_ref:]

    # Distance matrix
    if cfg.TEST.RE_RANKING:
        logger.info("Computing Re-ranking distance matrix...")
        distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    else:
        logger.info("Computing Euclidean distance matrix...")
        distmat = euclidean_distance(qf, gf)

    # Evaluate (only meaningful when ground-truth labels are available)
    if q_pids[0] != -1:
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        logger.info("Validation Results (TTA)")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    else:
        logger.info("Test set has no labels (competition mode) — skipping mAP evaluation.")

    logger.info("Total TTA inference time: {:.2f}s".format(time.time() - t0))
    return distmat


# ---------------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID TTA Inference")
    parser.add_argument(
        "--config_file", default="./config/UrbanElementsReID_test.yml",
        help="Path to config file", type=str
    )
    parser.add_argument(
        "--weight", default="", help="Path to model weights (.pth)", type=str
    )
    parser.add_argument(
        "opts", help="Modify config options via command line", default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Enable TTA
    cfg.defrost()
    cfg.TEST.TTA = True
    if args.weight:
        cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("PAT.test_tta", output_dir, if_train=False)
    logger.info("TTA scales: {}".format(cfg.TEST.TTA_SCALES))
    logger.info("RE_RANKING: {}".format(cfg.TEST.RE_RANKING))
    logger.info("Weight: {}".format(cfg.TEST.WEIGHT))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        logger.info("Dataset: {}".format(testname))
        do_inference_tta(cfg, model, testname)
