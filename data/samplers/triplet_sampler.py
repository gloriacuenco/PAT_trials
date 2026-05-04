import copy
import itertools
from collections import defaultdict
import random
from typing import Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from utils import comm


def no_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            # camid = info[2]
            camid = info[3]['domains']
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def _get_epoch_indices(self):
        # Shuffle identity list
        identities = np.random.permutation(self.num_identities)

        # If remaining identities cannot be enough for a batch,
        # just drop the remaining parts
        drop_indices = self.num_identities % self.num_pids_per_batch
        if drop_indices: identities = identities[:-drop_indices]

        ret = []
        for kid in identities:
            i = np.random.choice(self.pid_index[self.pids[kid]])
            i_cam = self.data_source[i][3]['domains']
            # _, i_pid, i_cam = self.data_source[i]
            ret.append(i)
            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = no_index(cams, i_cam)

            if select_cams:
                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                for kk in cam_indexes:
                    ret.append(index[kk])
            else:
                select_indexes = no_index(index, i)
                if not select_indexes:
                    # only one image for this identity
                    ind_indexes = [0] * (self.num_instances - 1)
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return ret

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, batch_size: int, num_instances: int, delete_rem: bool, seed: Optional[int] = None, cfg = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.delete_rem = delete_rem

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


        val_pid_index = [len(x) for x in self.pid_index.values()]
        min_v = min(val_pid_index)
        max_v = max(val_pid_index)
        hist_pid_index = [val_pid_index.count(x) for x in range(min_v, max_v+1)]
        num_print = 5
        for i, x in enumerate(range(min_v, min_v+min(len(hist_pid_index), num_print))):
            print('dataset histogram [bin:{}, cnt:{}]'.format(x, hist_pid_index[i]))
        print('...')
        print('dataset histogram [bin:{}, cnt:{}]'.format(max_v, val_pid_index.count(max_v)))

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                if self.delete_rem:
                    if x < self.num_instances:
                        val_pid_index_upper.append(x - v_remain + self.num_instances)
                    else:
                        val_pid_index_upper.append(x - v_remain)
                else:
                    val_pid_index_upper.append(x - v_remain + self.num_instances)

        total_images = sum(val_pid_index_upper)
        total_images = total_images - (total_images % self.batch_size) - self.batch_size # approax
        self.total_images = total_images



    def _get_epoch_indices(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_index[pid]) # whole index for each ID
            if self.delete_rem:
                if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            else:
                if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                elif (len(idxs) % self.num_instances) != 0:
                    idxs.extend(np.random.choice(idxs, size=self.num_instances - len(idxs) % self.num_instances, replace=False))

            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(int(idx))
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0: avai_pids.remove(pid)

        return final_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
        # return iter(self._get_epoch_indices())

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, info in enumerate(self.data_source):
            pid = info[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class DomainSuffleSampler(Sampler):

    def __init__(self, data_source: str, batch_size: int, num_instances: int, delete_rem: bool, seed: Optional[int] = None, cfg = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        self.delete_rem = delete_rem

        self.index_pid = defaultdict(list)
        self.pid_domain = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):

            domainid = info[3]['domains']
            if cfg.DATALOADER.CAMERA_TO_DOMAIN:
                pid = info[1] + str(domainid)
            else:
                pid = info[1]
            self.index_pid[index] = pid
            # self.pid_domain[pid].append(domainid)
            self.pid_domain[pid] = domainid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.domains = list(self.pid_domain.values())

        self.num_identities = len(self.pids)
        self.num_domains = len(set(self.domains))

        self.batch_size //= self.num_domains
        self.num_pids_per_batch //= self.num_domains

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()


        val_pid_index = [len(x) for x in self.pid_index.values()]
        min_v = min(val_pid_index)
        max_v = max(val_pid_index)
        hist_pid_index = [val_pid_index.count(x) for x in range(min_v, max_v+1)]
        num_print = 5
        for i, x in enumerate(range(min_v, min_v+min(len(hist_pid_index), num_print))):
            print('dataset histogram [bin:{}, cnt:{}]'.format(x, hist_pid_index[i]))
        print('...')
        print('dataset histogram [bin:{}, cnt:{}]'.format(max_v, val_pid_index.count(max_v)))

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                if self.delete_rem:
                    if x < self.num_instances:
                        val_pid_index_upper.append(x - v_remain + self.num_instances)
                    else:
                        val_pid_index_upper.append(x - v_remain)
                else:
                    val_pid_index_upper.append(x - v_remain + self.num_instances)

        cnt_domains = [0 for x in range(self.num_domains)]
        for val, index in zip(val_pid_index_upper, self.domains):
            cnt_domains[index] += val
        self.max_cnt_domains = max(cnt_domains)
        self.total_images = self.num_domains * (self.max_cnt_domains - (self.max_cnt_domains % self.batch_size) - self.batch_size)



    def _get_epoch_indices(self):


        def _get_batch_idxs(pids, pid_index, num_instances, delete_rem):
            batch_idxs_dict = defaultdict(list)
            for pid in pids:
                idxs = copy.deepcopy(pid_index[pid])
                if delete_rem:
                    if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                else:
                    if len(idxs) < self.num_instances: # if idxs is smaller than num_instance, choice redundantly
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                    elif (len(idxs) % self.num_instances) != 0:
                        idxs.extend(np.random.choice(idxs, size=self.num_instances - len(idxs) % self.num_instances, replace=False))

                np.random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(int(idx))
                    if len(batch_idxs) == num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            return batch_idxs_dict

        batch_idxs_dict = _get_batch_idxs(self.pids, self.pid_index, self.num_instances, self.delete_rem)

        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)

        local_avai_pids = \
            [[pids for pids, idx in zip(avai_pids, self.domains) if idx == i]
             for i in list(set(self.domains))]
        local_avai_pids_save = copy.deepcopy(local_avai_pids)


        revive_idx = [False for i in range(self.num_domains)]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch and not all(revive_idx):
            for i in range(self.num_domains):
                selected_pids = np.random.choice(local_avai_pids[i], self.num_pids_per_batch, replace=False)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
                        local_avai_pids[i].remove(pid)
            for i in range(self.num_domains):
                if len(local_avai_pids[i]) < self.num_pids_per_batch:
                    print('{} is recovered'.format(i))
                    batch_idxs_dict_new = _get_batch_idxs(self.pids, self.pid_index, self.num_instances, self.delete_rem)

                    revive_idx[i] = True
                    cnt = 0
                    for pid, val in batch_idxs_dict_new.items():
                        if self.domains[cnt] == i:
                            batch_idxs_dict[pid] = copy.deepcopy(batch_idxs_dict_new[pid])
                        cnt += 1
                    local_avai_pids[i] = copy.deepcopy(local_avai_pids_save[i])
                    avai_pids.extend(local_avai_pids_save[i])
                    avai_pids = list(set(avai_pids))
        return final_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            indices = self._get_epoch_indices()
            yield from indices


class MacroClassBalancedSampler(Sampler):
    """
    Sample a balanced number of identities from each macro class per batch.

    Each batch step contributes:
        pids_per_class * num_macro_classes * num_instances images

    where pids_per_class = (batch_size // num_instances) // num_macro_classes.

    This ensures the model sees an equal number of traffic signs, rubbish bins,
    containers, and crosswalks regardless of class imbalance in the dataset.

    Args:
        data_source: list of (img_path, pid, camid, others_dict) tuples.
            others_dict must contain 'macro_class' key.
        batch_size: total images per batch (IMS_PER_BATCH).
        num_instances: number of images per identity per batch.
        seed: random seed.
    """

    def __init__(self, data_source, batch_size: int, num_instances: int,
                 seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances

        # --- Build index structures ---
        # macro_class -> list of pids
        self.macro_class_pids = defaultdict(set)
        # pid -> list of dataset indices
        self.pid_index = defaultdict(list)

        for idx, info in enumerate(data_source):
            pid = info[1]
            others = info[3] if len(info) > 3 else {}
            macro_class = (others.get('macro_class', '') if isinstance(others, dict) else '')
            if not macro_class:
                macro_class = 'unknown'
            self.macro_class_pids[macro_class].add(pid)
            self.pid_index[pid].append(idx)

        # Sort for determinism
        self.macro_classes = sorted(self.macro_class_pids.keys())
        self.num_macro_classes = len(self.macro_classes)

        # Convert sets to sorted lists
        self.macro_class_pids = {
            cls: sorted(pids) for cls, pids in self.macro_class_pids.items()
        }

        # Number of identities to pick per class per batch step
        self.pids_per_class = max(1, (batch_size // num_instances) // self.num_macro_classes)
        self.effective_batch_size = self.pids_per_class * self.num_macro_classes * num_instances

        # Print summary
        print(f'\n[MacroClassBalancedSampler] {self.num_macro_classes} macro classes detected:')
        for cls in self.macro_classes:
            n_ids = len(self.macro_class_pids[cls])
            n_imgs = sum(len(self.pid_index[p]) for p in self.macro_class_pids[cls])
            print(f'  [{cls}]: {n_ids} identities, {n_imgs} images')
        print(f'  Sampling {self.pids_per_class} identities/class → '
              f'{self.effective_batch_size} images/batch (configured: {batch_size})\n')

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def _build_pool(self):
        """
        For each macro class, build a shuffled list of image-index groups.
        Each group is a list of `num_instances` image indices for one identity.
        """
        class_pools = {}
        for cls in self.macro_classes:
            pool = []
            for pid in self.macro_class_pids[cls]:
                idxs = copy.deepcopy(self.pid_index[pid])
                # Ensure we have at least num_instances indices
                if len(idxs) < self.num_instances:
                    idxs = list(np.random.choice(idxs, size=self.num_instances, replace=True))
                elif len(idxs) % self.num_instances != 0:
                    # Pad to a multiple of num_instances
                    extra = self.num_instances - (len(idxs) % self.num_instances)
                    idxs += list(np.random.choice(idxs, size=extra, replace=False))
                np.random.shuffle(idxs)
                # Chunk into groups of num_instances
                for i in range(0, len(idxs), self.num_instances):
                    pool.append(idxs[i:i + self.num_instances])
            np.random.shuffle(pool)
            class_pools[cls] = pool
        return class_pools

    def _get_epoch_indices(self):
        class_pools = self._build_pool()
        final_idxs = []

        while True:
            # Stop when any class has fewer groups than pids_per_class
            if not all(len(class_pools[cls]) >= self.pids_per_class
                       for cls in self.macro_classes):
                break
            # Sample pids_per_class groups from each class
            for cls in self.macro_classes:
                for _ in range(self.pids_per_class):
                    group = class_pools[cls].pop()
                    final_idxs.extend(group)

        return final_idxs

    def __iter__(self):
        """Return a finite iterator covering one epoch of balanced indices."""
        return iter(self._get_epoch_indices())

    def __len__(self):
        # Approximate: epoch length is limited by the smallest class
        min_class_batches = min(
            sum(len(self.pid_index[p]) for p in self.macro_class_pids[cls]) // self.num_instances
            for cls in self.macro_classes
        )
        return min_class_batches * self.num_macro_classes * self.num_instances
