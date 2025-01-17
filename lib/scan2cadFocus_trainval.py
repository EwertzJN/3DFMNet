import argparse
import os
import os.path as osp
import time
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from IPython import embed

from vision3d.engine import Engine
from vision3d.utils.metrics import Timer, StatisticsDictMeter
from vision3d.utils.torch_utils import to_cuda, all_reduce_dict
from vision3d.utils.python_utils import ensure_dir

from configs.config_scan2cad_focus import config
from utils.dataset import Process_Scan2cad_train_data_loader
from models.focusnet import create_model
from models.loss import FocusLoss, Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', metavar='N', type=int, default=10, help='iteration steps for logging')
    parser.add_argument('--output_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp1')
    parser.add_argument('--snapshot_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp1/snapshots')
    parser.add_argument('--logs_dir', type=str, default='/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp1/logs')
    parser.add_argument('--scan2cad_root', type=str, default='/data1/local_userdata/zhangliyuan/dataset/NIPS2024/Robi/')
    return parser


def run_one_epoch(
        engine,
        epoch,
        data_loader,
        model,
        evaluator,
        loss_func=None,
        optimizer=None,
        scheduler=None,
        training=True
):
    if training:
        model.train()
        if engine.distributed:
            data_loader.sampler.set_epoch(epoch)
    else:
        model.eval()

    
    timer = Timer()

    num_iter_per_epoch = len(data_loader)
    for i, data_dict in enumerate(data_loader):

        data_dict = to_cuda(data_dict)
        timer.add_prepare_time()

        if training:
            output_dict = model(data_dict)

            loss_dict = loss_func(output_dict, data_dict)
        else:
            with torch.no_grad():
                ref_length_c = data_dict['stack_lengths'][-1][0].item()                
                points_c = data_dict['points'][-1].detach()
                points_m = data_dict['points'][1].detach()
                
                ref_points_c = points_c[:ref_length_c]
                src_points_c = points_c[ref_length_c:]
                output_dict = model(data_dict)
                result_dict = evaluator(output_dict, data_dict)
                result_dict = {key: value for key, value in result_dict.items()}
                
        if training:
            loss = loss_dict['loss']

            if engine.distributed:
                loss_dict = all_reduce_dict(loss_dict, world_size=engine.world_size)
            loss_dict = {key: value.item() for key, value in loss_dict.items()}
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timer.add_process_time()

        if (i + 1) % engine.args.steps == 0:
            message = 'Epoch {}/{}, '.format(epoch + 1, config.max_epoch) + \
                    'iter {}/{}, '.format(i + 1, num_iter_per_epoch)
            if training:
                message += 'loss: {:.3f}, '.format(loss_dict['loss']) + \
                        'c_loss: {:.3f}, '.format(loss_dict['c_loss']) + \
                            'center_norm_loss: {:.3f}, '.format(loss_dict['center_norm_loss']) + \
                            'center_dir_loss: {:.3f}, '.format(loss_dict['center_dir_loss']) + \
                        'instance_loss: {:.3f}, '.format(loss_dict['instance_loss'])# + \
                        # 'center_loss: {:.3f},'.format(loss_dict['center_loss'])
            else:
                message += 'precision: {:.3f}, '.format(result_dict['precision']) + \
                    'recall: {:.3f}, '.format(result_dict['recall']) + \
                    'F1_score: {:.3f}, '.format(result_dict['F1_score']) 
                            
            if training:
                message += 'lr: {:.3e}, '.format(scheduler.get_last_lr()[0])
            message += 'time: {:.3f}s/{:.3f}s'.format(timer.get_prepare_time(), timer.get_process_time())
            if not training:
                message = '[Eval] ' + message
            engine.logger.info(message)

        if training:
            engine.step()
        torch.cuda.empty_cache()


    message = 'Epoch {}, '.format(epoch + 1)
 
    if not training:
        message = '[Eval] ' + message
    engine.logger.critical(message)

    if training:
        engine.register_state(epoch=epoch)
        if engine.local_rank == 0:
            snapshot = osp.join(engine.args.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
            engine.save_snapshot(snapshot)
        scheduler.step()


def main():
    parser = make_parser()
    args = parser.parse_args()
    config.process_scan2cad_root = args.scan2cad_root
    ensure_dir(args.output_dir)
    ensure_dir(args.snapshot_dir)
    ensure_dir(args.logs_dir)
    log_file = osp.join(args.logs_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        start_time = time.time()
        train_loader, neighborhood_limits = Process_Scan2cad_train_data_loader(engine, config)
        loading_time = time.time() - start_time
        message = 'Neighborhood limits: {}.'.format(neighborhood_limits)
        engine.logger.info(message)
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        if engine.distributed:
            local_rank = engine.local_rank
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = optim.Adam(model.parameters(),
                               lr=config.learning_rate * engine.world_size,
                               weight_decay=config.weight_decay)
        loss_func = FocusLoss(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model, optimizer=optimizer)
        engine.args.snapshot=None
        if engine.args.snapshot is not None:
            engine.load_snapshot(engine.args.snapshot)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.gamma, last_epoch=engine.state.epoch)

        for epoch in range(engine.state.epoch + 1, config.max_epoch):
            run_one_epoch(
                engine, epoch, train_loader, model, evaluator, loss_func=loss_func, optimizer=optimizer,
                scheduler=scheduler, training=True
            )
            """ run_one_epoch(
                engine, epoch, valid_loader, model, evaluator, training=False
            ) """


if __name__ == '__main__':
    main()
