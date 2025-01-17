import random

import torch
from torch.autograd import Function

from ... import ext


__all__ = [
    'gather',
    'group_gather',
    'farthest_point_sampling',
    'farthest_point_sampling_and_gather',
    'ball_query',
    'random_ball_query',
    'k_nearest_neighbors',
    'dilated_k_nearest_neighbors',
    'batched_pairwise_distance'
]


class _GatherFunction(Function):
    r"""
    Gather by indices.
    Gather `num_sample` points from a set of `num_point` points.

    :param points: torch.Tensor (batch_size, num_channel, num_point)
        The features/coordinates of all points.
    :param indices: torch.Tensor (batch_size, num_sample)
        The indices of the points to gather.

    :return output: torch.Tensor (batch, num_channel, num_sample)
        The features/coordinates of the gathered points.
    """
    @staticmethod
    def forward(ctx, points, indices):
        batch_size, num_channel, num_point = points.shape
        ctx.save_for_backward(indices)
        ctx.num_point = num_point
        output = ext.gather_by_index(points, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]
        num_point = ctx.num_point
        grad_output = grad_output.contiguous()
        grad_points = ext.gather_by_index_grad(grad_output, indices, num_point)
        return grad_points, None


gather = _GatherFunction.apply


class _GroupGatherFunction(Function):
    @staticmethod
    def forward(ctx, points, indices):
        r"""
        Group gather by indices.
        Gather `num_sample` points for each of the `num_centroid` centroids from a set of `num_point` points.

        :param points: torch.Tensor (batch_size, num_channel, num_point)
            The features/coordinates of all points.
        :param indices: torch.Tensor (batch_size, num_centroid, num_sample)
            The indices of the points to gather.

        :return output: torch.Tensor (batch, num_channel, num_centroid, num_sample)
            The features/coordinates of the gathered points.
        """
        batch_size, num_channel, num_point = points.shape
        ctx.save_for_backward(indices)
        ctx.num_point = num_point
        output = ext.group_gather_by_index(points, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]
        num_point = ctx.num_point
        grad_output = grad_output.contiguous()
        grad_points = ext.group_gather_by_index_grad(grad_output, indices, num_point)
        return grad_points, None


group_gather = _GroupGatherFunction.apply


def farthest_point_sampling(points, num_sample):
    r"""
    Farthest point sampling.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param num_sample: int
        The number of centroids sampled.
    :return indices: torch.Tensor (batch_size, num_sample)
        The indices of the sampled centroids.
    """
    indices = ext.farthest_point_sampling(points, num_sample)
    return indices


def farthest_point_sampling_and_gather(points, num_sample):
    r"""
    Farthest point sampling and gather the coordinates of the sampled points.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param num_sample: int
        The number of centroids sampled.
    :return samples: torch.Tensor (batch_size, 3, num_sample)
        The indices of the sampled points.
    """
    indices = farthest_point_sampling(points, num_sample)
    samples = gather(points, indices)
    return samples


def random_point_sampling(points, num_sample):
    device = points.device
    batch_size, _, num_point = points.shape
    weights = torch.ones(batch_size, num_point).to(device)
    indices = torch.multinomial(weights, num_sample)
    return indices


def random_point_sampling_and_gather(points, num_sample):
    indices = random_point_sampling(points, num_sample)
    samples = gather(points, indices)
    return samples


def ball_query(points, centroids, num_sample, radius):
    r"""
    Ball query without random sampling.
    The first `num_sample` points in the ball for each centroids are returned.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param centroids: torch.Tensor (batch_size, 3, num_centroid)
        The coordinates of the centroids.
    :param num_sample: int
        The number of points sampled for each centroids.
    :param radius: float
        The radius of the balls.

    :return indices: torch.Tensor (batch_size, num_centroid, num_sample)
        The indices of the sampled points.
    """
    indices = ext.ball_query_v1(centroids, points, radius, num_sample)
    return indices


def random_ball_query(points, centroids, num_sample, radius):
    r"""
    Ball query with random sampling.
    All points in the ball are found, then `num_sample` points are randomly sampled as the results.

    :param points: torch.Tensor (batch_size, 3, num_point)
        The coordinates of the points.
    :param centroids: torch.Tensor (batch_size, 3, num_centroid)
        The coordinates of the centroids.
    :param num_sample: int
        The number of points sampled for each centroids.
    :param radius: float
        The radius of the balls.

    :return indices: torch.Tensor (batch_size, num_centroid, num_sample)
        The indices of the sampled points.
    """
    # seed = random.randint()
    seed = 3075
    indices = ext.ball_query_v2(seed, centroids, points, radius, num_sample)
    return indices


def three_nearest_neighbors(points1, points2):
    r"""
    Compute the three nearest neighbors for the non-sampled points in the sub-sampled points.

    :param points1: torch.Tensor (batch_size, 3, num_point1)
        The points to compute the three nearest neighbors.
    :param points2: torch.Tensor (batch_size, 3, num_point2)
        The sub-sampled points set.
    :return dist2: torch.Tensor (batch_size, num_point1, 3)
        The corresponding squared distance of the found neighbors to the point.
    :return indices: torch.Tensor (batch_size, num_point1, 3)
        The indices of the found neighbors.
    """
    dist2, indices = ext.three_nearest_neighbors(points1, points2)
    return dist2, indices


class _ThreeInterpolateFunction(Function):
    r"""
    Interpolate the features for the non-sampled points from the sub-sampled points.
    Three sub-sampled points are used to interpolate one non-sampled point.

    :param features: torch.Tensor (batch_size, num_channel, num_point2)
        The features of the sub-sampled points.
    :param indices: torch.Tensor (batch, num_point1, 3)
        The indices of the sub-sampled points to interpolate each non-sampled points.
    :param weight: torch.Tensor (batch, num_point1, 3)
        The weights of the sub-sampled points to interpolate each non-sampled points.
    :return outputs: torch.Tensor (batch_size, num_channel, num_point1)
        The interpolated features of the non-sampled points.
    """
    @staticmethod
    def forward(ctx, features, indices, weights):
        batch_size, num_channel, num_point2 = features.shape
        ctx.save_for_backward(indices, weights)
        ctx.num_point2 = num_point2
        outputs = ext.three_interpolate(features, indices, weights)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        indices, weights = ctx.saved_tensors
        num_point2 = ctx.num_point2
        grad_outputs = grad_outputs.contiguous()
        grad_features = ext.three_interpolate_grad(grad_outputs, indices, weights, num_point2)
        return grad_features, None, None


three_interpolate = _ThreeInterpolateFunction.apply


def batched_pairwise_distance(points0, points1, normalized=False, clamp=False):
    r"""
    Compute the squared pairwise distance of two batched point clouds.

    :param points0: torch.Tensor (batch_size, num_feature, num_point0)
        The features/coordinates of the first batched point cloud.
    :param points1: torch.Tensor (batch_size, num_feature, num_point1)
        The features/coordinates of the second batched point cloud.
    :param normalized: bool (default: False)
        Whether the inputs are normalized.
    :return pairwise_dist2: torch.Tensor (batch_size, num_point0, num_point1)
        The squared pairwise distance of the two batched point clouds.
    """
    ab = torch.matmul(points0.transpose(1, 2), points1)
    if normalized:
        pairwise_dist2 = 2 - 2 * ab
    else:
        a2 = torch.sum(points0 ** 2, dim=1).unsqueeze(2)
        b2 = torch.sum(points1 ** 2, dim=1).unsqueeze(1)
        pairwise_dist2 = a2 - 2 * ab + b2
    if clamp:
        pairwise_dist2 = torch.maximum(pairwise_dist2, torch.zeros_like(pairwise_dist2))
    return pairwise_dist2


def k_nearest_neighbors(points, centroids, num_neighbor, return_distance=False, remove_nearest=False):
    r"""
    Compute the kNNs of the points in `centroids` from the points in `points`.

    Note: This implementation decomposes uses less memory than the naive implementation:
    `pairwise_dist2 = torch.sum((centroids.unsqueeze(3) - points.unsqueeze(2)) ** 2, dim=1)`

    :param points: torch.Tensor (batch_size, num_feature, num_point)
        The features/coordinates of the points from which the kNNs are computed.
    :param centroids: torch.Tensor (batch_size, num_feature, num_centroid)
        The features/coordinates of the centroid points whose kNNs are computed.
    :param num_neighbor: int
        The number of nearest neighbors to compute.
    :param return_distance: bool (default: False)
        Whether return distances.
    :param remove_nearest: bool (default: True)
        Whether remove the nearest neighbor (in case where points and centroids are the same).
    :return indices: torch.Tensor(batch_size, num_points1, k)
        The indices of the kNNs of the centroids.
    """
    square_pairwise_distances = batched_pairwise_distance(centroids, points)

    if remove_nearest:
        square_distances, indices = square_pairwise_distances.topk(num_neighbor + 1, dim=2, largest=False)
        square_distances = square_distances[:, :, 1:]
        indices = indices[:, :, 1:]
    else:
        square_distances, indices = square_pairwise_distances.topk(num_neighbor, dim=2, largest=False)

    if return_distance:
        return square_distances, indices
    else:
        return indices


def dilated_k_nearest_neighbors(points, centroids, num_neighbor, dilation, remove_nearest=False):
    num_neighbor_dilated = num_neighbor * dilation
    indices = k_nearest_neighbors(points, centroids, num_neighbor_dilated, remove_nearest=remove_nearest)
    indices = indices[:, :, ::dilation].contiguous()
    return indices
