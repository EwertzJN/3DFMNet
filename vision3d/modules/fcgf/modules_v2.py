import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed

from ...utils.point_cloud_utils import pairwise_distance


class HardestContrastiveLossGpu(nn.Module):
    def __init__(self, pos_radius, neg_radius, pos_margin, neg_margin, max_correspondence, num_candidate):
        super(HardestContrastiveLossGpu, self).__init__()
        self.pos_radius = pos_radius
        self.neg_radius = neg_radius
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.max_correspondence = max_correspondence
        self.num_candidate = num_candidate

    def forward(self, points0, points1, feats0, feats1, correspondences):
        NotImplemented
