import MinkowskiEngine as ME
import numpy as np


def sparse_quantize(points, voxel_size, feats=None, return_index=False):
    indices = ME.utils.sparse_quantize(points, quantization_size=voxel_size, return_index=True)[1]
    points = points[indices]
    if feats is not None:
        feats = feats[indices]
    coords = np.floor(points / voxel_size).astype(np.int32)

    outputs = [points]
    if feats is not None:
        outputs.append(feats)
    outputs.append(coords)
    if return_index:
        outputs.append(indices)

    return outputs
