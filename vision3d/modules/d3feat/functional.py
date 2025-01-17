import torch
import torch.nn.functional as F
from IPython import embed


def detection_scores(features, batch, training):
    neighbors = batch['neighbors'][0]  # [n_points, n_neighbors]
    length0, length1 = batch['stack_lengths'][0]
    total_length = length0 + length1

    # add a fake point in the last row for shadow neighbors
    shadow_features = torch.zeros_like(features[:1, :])
    features = torch.cat([features, shadow_features], dim=0)
    shadow_neighbor = torch.ones_like(neighbors[:1, :]) * total_length
    neighbors = torch.cat([neighbors, shadow_neighbor], dim=0)

    # #  normalize the feature to avoid overflow
    features = features / (torch.max(features) + 1e-6)

    # local max score (saliency score)
    neighbor_features = features[neighbors, :]  # [n_points, n_neighbors, 64]
    neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
    neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdims=True)  # [n_points, 1]
    neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
    mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num  # [n_points, 64]
    local_max_score = F.softplus(features - mean_features)  # [n_points, 64]

    # calculate the depth-wise max score
    depth_wise_max = torch.max(features, dim=1, keepdims=True)[0]  # [n_points, 1]
    depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, 64]

    all_scores = local_max_score * depth_wise_max_score
    # use the max score among channel to be the score of a single point.
    scores = torch.max(all_scores, dim=1)[0]  # [n_points]

    # hard selection (used during test)
    if not training:
        local_max = torch.max(neighbor_features, dim=1)[0]
        is_local_max = (features == local_max).float()
        # print(f"Local Max Num: {float(is_local_max.sum().detach().cpu())}")
        detected = torch.max(is_local_max, dim=1)[0]
        scores = scores * detected

    return scores[:-1]
