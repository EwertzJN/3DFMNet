import torch
import torch.nn as nn


@torch.no_grad()
def nms(scores, suppression_factor_map, num_sel=None):
    r"""
    [PyTorch] Non-Maximum Suppression with a pre-computed suppression factor map.

    :param scores: torch.Tensor (N,)
    :param suppression_factor_map: torch.Tensor (N, N)
        Note: the diagonal of `suppression_factor_map` should be zeros.
    :param num_sel: int or None
    :return suppressed_scores: torch.Tensor (N,)
    """
    num_item = scores.shape[0]
    suppressed_scores = torch.zeros_like(scores)
    if num_sel is None:
        num_sel = num_item
    for i in range(num_sel):
        max_scores, maxi = torch.max(scores, dim=0)
        suppressed_scores[maxi] = max_scores
        suppression_factors = suppression_factor_map[maxi]
        scores = scores * suppression_factors
    return suppressed_scores
