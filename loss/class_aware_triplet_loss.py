import torch
from torch import nn

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

class ClassAwareTripletLoss(nn.Module):
    def __init__(self, margin1=0.3, margin2=0.1):
        super(ClassAwareTripletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.ranking_loss = nn.MarginRankingLoss(margin=margin1)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2)

    def forward(self, inputs, targets, macro_classes):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size) - micro-identity
            macro_classes: list or tensor of macro class strings/IDs
        """
        n = inputs.size(0)
        
        # compute euclidean distance
        dist = euclidean_dist(inputs, inputs)
        
        # For each anchor, find the hardest positive and hardest negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        # Create mask for same macro_class
        if isinstance(macro_classes[0], str):
            macro_classes = [c.lower() for c in macro_classes]
            macro_mask = torch.tensor([[c1 == c2 for c2 in macro_classes] for c1 in macro_classes], device=inputs.device)
        else:
            macro_mask = macro_classes.expand(n, n).eq(macro_classes.expand(n, n).t())
            
        dist_ap, dist_an = [], []
        dist_an_inter = [] # inter-class negatives (different macro class)
        
        for i in range(n):
            # hardest positive (same micro-identity)
            pos_mask = mask[i]
            dist_ap.append(dist[i][pos_mask].max().unsqueeze(0))
            
            # negatives of the SAME macro-class (intra-class negatives)
            neg_mask_intra = (macro_mask[i]) & (~mask[i])
            if neg_mask_intra.sum() > 0:
                dist_an.append(dist[i][neg_mask_intra].min().unsqueeze(0))
            else:
                dist_an.append(dist[i][~mask[i]].min().unsqueeze(0)) # fallback
                
            # negatives of a DIFFERENT macro-class (inter-class negatives)
            neg_mask_inter = ~macro_mask[i]
            if neg_mask_inter.sum() > 0:
                dist_an_inter.append(dist[i][neg_mask_inter].min().unsqueeze(0))
            else:
                dist_an_inter.append(dist[i][~mask[i]].min().unsqueeze(0)) # fallback
                
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_an_inter = torch.cat(dist_an_inter)
        
        # Compute ranking losses
        y = torch.ones_like(dist_an)
        
        # Tier 1: Intra-Class Margin
        loss1 = self.ranking_loss(dist_an, dist_ap, y)
        
        # Tier 2: Inter-Class Margin
        loss2 = self.ranking_loss2(dist_an_inter, dist_ap, y)
        
        loss = loss1 + loss2
        return loss, dist_ap, dist_an
