import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed

from ..geometry.functional import k_nearest_neighbors, group_gather, farthest_point_sampling, gather
from ...utils.torch_utils import create_conv1d_blocks, create_conv2d_blocks


class PointTransformerLayer(nn.Module):
    def __init__(self, feature_dim):
        super(PointTransformerLayer, self).__init__()
        self.feature_dim = feature_dim
        layers = create_conv2d_blocks(3, [feature_dim, feature_dim], 1, batch_norm=False, activation=None)
        layers.append(('relu', nn.ReLU()))
        self.position_encoding = nn.Sequential(OrderedDict(layers))
        self.q_layer = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.k_layer = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.v_layer = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        layers = create_conv2d_blocks(feature_dim, [feature_dim, feature_dim], 1, batch_norm=False, activation=None)
        layers.append(('relu', nn.ReLU()))
        self.attention_encoding = nn.Sequential(OrderedDict(layers))

    def forward(self, feats, grouped_feats, points, grouped_points):
        feats = feats.unsqueeze(3)
        points = points.unsqueeze(3)
        delta = self.position_encoding(points - grouped_points)

        K = self.k_layer(feats)
        Q = self.q_layer(grouped_feats)
        V = self.v_layer(grouped_feats) + delta
        attention_scores = self.attention_encoding(K - Q + delta)
        attention_scores = F.softmax(attention_scores, dim=3)
        output = torch.sum(attention_scores * V, dim=3)

        return output


class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_neighbor):
        super(PointTransformerBlock, self).__init__()
        self.num_neighbor = num_neighbor
        self.r_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.e_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.point_transformer = PointTransformerLayer(hidden_dim)

    def forward(self, feats, points):
        identity = feats
        feats = self.r_layer(feats)
        indices = k_nearest_neighbors(points, points, self.num_neighbor)
        grouped_feats = group_gather(feats, indices)
        grouped_points = group_gather(points, indices)
        feats = self.point_transformer(feats, grouped_feats, points, grouped_points)
        feats = self.e_layer(feats)
        feats = feats + identity
        return feats, points


class TransitionDownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, downsampling_ratio, num_neighbor):
        super(TransitionDownBlock, self).__init__()
        self.downsampling_ratio = downsampling_ratio
        self.num_neighbor = num_neighbor
        layers = create_conv1d_blocks(input_dim, [output_dim, output_dim], kernel_size=1)
        self.transition_layer = nn.Sequential(OrderedDict(layers))

    def forward(self, feats, points):
        feats = self.transition_layer(feats)
        num_sample = int(np.ceil(points.shape[2] / self.downsampling_ratio))
        indices = farthest_point_sampling(points, num_sample)
        centroids = gather(points, indices)
        indices = k_nearest_neighbors(points, centroids, self.num_neighbor)
        grouped_feats = group_gather(feats, indices)
        feats = grouped_feats.mean(dim=3)
        return feats, centroids
