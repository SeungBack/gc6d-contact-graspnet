import warnings
warnings.filterwarnings(action='ignore')
import torch
import torch.nn as nn
import os
import json
from datetime import datetime, timedelta
from utils import builder, misc
import time
from utils.logger import *
from utils.metrics import Metrics
from utils.meters import AverageMeter

from utils.utils import send_dict_to_device
#* this is single-dual branch using ACRONYM dataset


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    # _, test_dataloader = builder.dataset_builder(args, config.dataset.test) # !shsh
    
    # build model
    config.model.arg_configs = getattr(args, 'arg_configs', []) or []
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(torch.device('cuda'))
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None
    
    if args.resume: 
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)
        
    # print model info
    print_log('Trainable_parameters:', logger=logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
    
    # train and val
    # for training
    base_model.zero_grad()
    total_epochs = config.max_epoch - start_epoch + 1
    total_steps = total_epochs * len(train_dataloader)
    training_start_time = time.time()
    for epoch in range(start_epoch, config.max_epoch+1):

        base_model.train()
        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['ContactGraspNetLoss'])
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            send_dict_to_device(data)
            data_time.update(time.time() - batch_start_time)
            pc = data['pc']
            
            optimizer.zero_grad()
            pred = base_model(pc)
            loss, bin_ce_loss, width_loss, adds_loss = base_model.get_loss(pred, data)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)

            optimizer.step()
            losses.update(loss.item())
            
            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/recon_loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/bin_ce_loss', bin_ce_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/width_loss', width_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/adds_loss', adds_loss.item(), n_itr)
            
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                steps_done = ((epoch - start_epoch) * n_batches) + idx + 1
                elapsed_time = time.time() - training_start_time
                avg_step_time = elapsed_time / max(steps_done, 1)
                remaining_steps = max(total_steps - steps_done, 0)
                eta_seconds = avg_step_time * remaining_steps
                eta_finish = datetime.now() + timedelta(seconds=eta_seconds)
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) ETA = %s FinishAt = %s Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            _format_duration(eta_seconds), eta_finish.strftime('%Y-%m-%d %H:%M:%S'),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            if n_itr % 20000 == 0 and n_itr !=0:
               builder.save_checkpoint(base_model, optimizer, epoch, metrics=None, best_metrics=None, prefix=f'ckpt-iter-{n_itr}', args=args, logger = logger)
            
        if isinstance(scheduler, list):
            for item in scheduler:                
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        builder.save_checkpoint(base_model, optimizer, epoch, metrics=None, best_metrics=None, prefix='ckpt-last', args=args, logger=logger)

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/recon_loss', losses.avg(), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        
    builder.save_checkpoint(base_model, optimizer, epoch, metrics=None, best_metrics=None, prefix=f'ckpt-final', args=args, logger = logger)

    if train_writer is not None:
        train_writer.close()
                    
