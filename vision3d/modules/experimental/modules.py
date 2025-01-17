from collections import OrderedDict

import torch
import torch.nn as nn

from ...utils.torch_utils import create_conv1d_blocks, create_conv2d_blocks
from .. import geometry
from .. import pointnet2


class RefinedSampling(nn.Module):
    def __init__(self, input_dim, output_dims1, output_dims2, num_centroid, num_sample, radius):
        super(RefinedSampling, self).__init__()

        self.num_centroid = num_centroid
        self.num_sample = num_sample
        self.radius = radius

        layers = create_conv2d_blocks(input_dim, output_dims1, kernel_size=1)
        self.pointnet1 = nn.Sequential(OrderedDict(layers))

        output_dims2.append(3)
        layers = create_conv1d_blocks(output_dims1[-1], output_dims2, kernel_size=1)
        self.pointnet2 = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features):
        centroids = geometry.functional.farthest_point_sampling_and_gather(points, self.num_centroid)
        features = pointnet2.functional.ball_query_and_group_gather(points, features, centroids, self.num_sample,
                                                                    self.radius)
        features = self.pointnet1(features)
        features, _ = features.max(dim=3)
        offsets = self.pointnet2(features)
        centroids = centroids + offsets
        return centroids


class RefinedSetAbstractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dims1, hidden_dims2, output_dims, num_centroid, num_sample, radius):
        super(RefinedSetAbstractionModule, self).__init__()
        self.num_centroid = num_centroid
        self.num_sample = num_sample
        self.radius = radius
        self.sampling = RefinedSampling(input_dim, hidden_dims1, hidden_dims2, num_centroid, num_sample, radius)
        layers = create_conv2d_blocks(input_dim, output_dims, kernel_size=1)
        self.pointnet = nn.Sequential(OrderedDict(layers))

    def forward(self, points, features):
        centroids = self.sampling(points, features)
        features = pointnet2.functional.ball_query_and_group_gather(points, features, centroids, self.num_sample,
                                                                    self.radius)
        features = self.pointnet(features)
        features, _ = features.max(dim=3)
        return centroids, features
