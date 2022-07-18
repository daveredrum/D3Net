import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxRankingLoss(nn.Module):
    def __init__(self, is_reduce=True):
        super().__init__()

        self.is_reduce = is_reduce

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=1)

        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1)

        # reduction
        if self.is_reduce:
            loss = loss.mean()

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, score, label):
        score *= self.gamma
        sim = (score*label).sum()
        neg_sim = score*label.logical_not()
        neg_sim = torch.logsumexp(neg_sim, dim=0) # soft max
        loss = torch.clamp(neg_sim - sim + self.margin, min=0).sum()

        return loss