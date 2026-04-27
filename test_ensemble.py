import os
import time
import argparse
import torch
import torch.nn as nn
from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval

def do_inference_ensemble(cfg1, cfg2, model1, model2, val_loader1, val_loader2, num_query):
    device = "cuda"
    logger = setup_logger("PAT.test", cfg1.LOG_ROOT, if_train=False)
    logger.info("Enter ensembling inference")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg1.TEST.FEAT_NORM, reranking=cfg1.TEST.RE_RANKING)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model1 = nn.DataParallel(model1)
            model2 = nn.DataParallel(model2)
        model1.to(device)
        model2.to(device)

    model1.eval()
    model2.eval()
    
    t0 = time.time()
    
    for (batch1, batch2) in zip(val_loader1, val_loader2):
        img1 = batch1['images']
        img2 = batch2['images']
        pid = batch1['targets']
        camids = batch1['camid']
        
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Model 1 features
            feat1 = model1(img1)
            if isinstance(feat1, tuple) or isinstance(feat1, list):
                feat1 = feat1[-1] if isinstance(feat1, list) else feat1[0]
                
            # Model 2 features
            feat2 = model2(img2)
            if isinstance(feat2, tuple) or isinstance(feat2, list):
                feat2 = feat2[-1] if isinstance(feat2, list) else feat2[0]
                
            # Concatenate features
            feat = torch.cat([feat1, feat2], dim=1)
            
            evaluator.update((feat, pid, camids))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Ensemble Testing")
    parser.add_argument("--config_file1", required=True, help="path to config file for Model 1", type=str)
    parser.add_argument("--weight1", required=True, help="path to weights for Model 1", type=str)
    parser.add_argument("--config_file2", required=True, help="path to config file for Model 2", type=str)
    parser.add_argument("--weight2", required=True, help="path to weights for Model 2", type=str)
    
    args = parser.parse_args()

    # Load Config 1
    cfg1 = cfg.clone()
    cfg1.merge_from_file(args.config_file1)
    cfg1.TEST.WEIGHT = args.weight1
    cfg1.freeze()

    # Load Config 2
    cfg2 = cfg.clone()
    cfg2.merge_from_file(args.config_file2)
    cfg2.TEST.WEIGHT = args.weight2
    cfg2.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg1.MODEL.DEVICE_ID

    print("Building Model 1...")
    model1 = make_model(cfg1, cfg1.MODEL.NAME, 0, 0, 0)
    model1.load_param(cfg1.TEST.WEIGHT)

    print("Building Model 2...")
    model2 = make_model(cfg2, cfg2.MODEL.NAME, 0, 0, 0)
    model2.load_param(cfg2.TEST.WEIGHT)

    for testname in cfg1.DATASETS.TEST:
        print(f"Building dataloaders for {testname}...")
        val_loader1, num_query1 = build_reid_test_loader(cfg1, testname)
        val_loader2, num_query2 = build_reid_test_loader(cfg2, testname)
        
        assert num_query1 == num_query2, "Mismatch in query sizes between loaders!"
        
        do_inference_ensemble(cfg1, cfg2, model1, model2, val_loader1, val_loader2, num_query1)
