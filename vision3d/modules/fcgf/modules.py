import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed

from ...utils.point_cloud_utils import pairwise_distance


def _hash(pairs, hash_seed):
    hash_vec = pairs[:, 0] + pairs[:, 1] * hash_seed
    return hash_vec


class HardestContrastiveLoss(nn.Module):
    def __init__(self, positive_thresh, negative_thresh, num_positive_sample, num_negative_candidate):
        super().__init__()
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh
        self.num_positive_sample = num_positive_sample
        self.num_negative_candidate = num_negative_candidate

    def forward(self, feats0, feats1, positive_pairs):
        num_point0 = feats0.shape[0]
        num_point1 = feats1.shape[1]
        num_positive_pair = positive_pairs.shape[0]
        hash_seed = max(num_point0, num_point1)
        num_candidate0 = min(num_point0, self.num_negative_candidate)
        num_candidate1 = min(num_point1, self.num_negative_candidate)
        candidate_indices0 = np.random.choice(num_point0, num_candidate0, replace=False)
        candidate_indices1 = np.random.choice(num_point1, num_candidate1, replace=False)

        if self.num_positive_sample < num_positive_pair:
            positive_pair_indices = np.random.choice(num_positive_pair, self.num_positive_sample, replace=False)
            sampled_positive_pairs = positive_pairs[positive_pair_indices]
        else:
            sampled_positive_pairs = positive_pairs

        candidate_feats0, candidate_feats1 = feats0[candidate_indices0], feats1[candidate_indices1]

        positive_indices0 = sampled_positive_pairs[:, 0].long()
        positive_indices1 = sampled_positive_pairs[:, 1].long()
        positive_feats0, positive_feats1 = feats0[positive_indices0], feats1[positive_indices1]

        distance0 = pairwise_distance(positive_feats0, candidate_feats1)
        distance1 = pairwise_distance(positive_feats1, candidate_feats0)

        negative_distance0, negative_indices0 = distance0.min(1)
        negative_distance1, negative_indices1 = distance1.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        positive_keys = _hash(positive_pairs, hash_seed)

        negative_indices0 = candidate_indices1[negative_indices0.detach().cpu().numpy()]
        negative_indices1 = candidate_indices0[negative_indices1.detach().cpu().numpy()]
        negative_pairs0 = np.stack([positive_indices0.numpy(), negative_indices0], axis=1)
        negative_pairs1 = np.stack([negative_indices1, positive_indices1.numpy()], axis=1)
        negative_keys0 = _hash(negative_pairs0, hash_seed)
        negative_keys1 = _hash(negative_pairs1, hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(negative_keys0, positive_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(negative_keys1, positive_keys, assume_unique=False)))
        positive_loss = F.relu((positive_feats0 - positive_feats1).pow(2).sum(1) - self.positive_thresh).mean()
        negative_loss0 = F.relu(self.negative_thresh - negative_distance0[mask0]).pow(2).mean()
        negative_loss1 = F.relu(self.negative_thresh - negative_distance1[mask1]).pow(2).mean()
        negative_loss = (negative_loss0 + negative_loss1) / 2
        total_loss = positive_loss + negative_loss

        result_dict = {}
        result_dict['loss'] = total_loss
        result_dict['pos_loss'] = positive_loss
        result_dict['neg_loss'] = negative_loss

        return result_dict
