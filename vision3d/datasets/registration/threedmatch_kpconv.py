from functools import partial

import torch
import torch.utils.data
import numpy as np

from ...modules.kpconv.helpers import generate_input_data, calibrate_neighbors


def threedmatch_kpconv_collate_fn(data_dicts, config, neighborhood_limits):
    new_data_dicts = []

    for data_dict in data_dicts:
        new_data_dict = {}

        new_data_dict['scene_name'] = data_dict['scene_name']
        new_data_dict['ref_frame'] = data_dict['ref_frame']
        new_data_dict['src_frame'] = data_dict['src_frame']
        new_data_dict['overlap'] = data_dict['overlap']
        new_data_dict['transform'] = torch.from_numpy(data_dict['transform'])
        if 'corr_indices' in data_dict:
            new_data_dict['corr_indices'] = torch.from_numpy(data_dict['corr_indices'])

        feats = np.concatenate([data_dict['ref_feats'], data_dict['src_feats']], axis=0)
        new_data_dict['features'] = torch.from_numpy(feats)

        ref_points, src_points = data_dict['ref_points'], data_dict['src_points']
        points = np.concatenate([ref_points, src_points], axis=0)
        lengths = np.array([ref_points.shape[0], src_points.shape[0]])
        stacked_points = torch.from_numpy(points)
        stacked_lengths = torch.from_numpy(lengths)

        input_points, input_neighbors, input_pools, input_upsamples, input_lengths = generate_input_data(
            stacked_points, stacked_lengths, config, neighborhood_limits
        )

        new_data_dict['points'] = input_points
        new_data_dict['neighbors'] = input_neighbors
        new_data_dict['pools'] = input_pools
        new_data_dict['upsamples'] = input_upsamples
        new_data_dict['stack_lengths'] = input_lengths

        new_data_dicts.append(new_data_dict)

    if len(new_data_dicts) == 1:
        return new_data_dicts[0]
    else:
        return new_data_dicts


def get_dataloader(
        dataset,
        config,
        batch_size,
        num_workers,
        shuffle=False,
        neighborhood_limits=None,
        drop_last=True,
        sampler=None
):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=threedmatch_kpconv_collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=partial(threedmatch_kpconv_collate_fn, config=config, neighborhood_limits=neighborhood_limits),
        drop_last=drop_last
    )
    return dataloader, neighborhood_limits
