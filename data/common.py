import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            pids = list()
            for i, item in enumerate(img_items):
                if item[1] in pids: continue
                pids.append(item[1])
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        item = self.img_items[index]
        img_path = item[0]
        pid = item[1]
        camid = item[2]
        others = item[3] if len(item) > 3 else ''
        img = read_image(img_path)
        
        # Custom logic interceptor for horizontal flip
        from config import cfg
        import random
        from PIL import Image
        if cfg.INPUT.DO_FLIP:
            skip_flip = False
            if isinstance(others, dict) and 'macro_class' in others:
                cls_name = others['macro_class'].lower()
                if 'trafficsign' in cls_name or 'trafficsignal' in cls_name:
                    skip_flip = True
            
            if not skip_flip and random.random() < cfg.INPUT.FLIP_PROB:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
        if self.transform is not None: img = self.transform(img)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "others": others
        }

    @property
    def num_classes(self):
        return len(self.pids)
