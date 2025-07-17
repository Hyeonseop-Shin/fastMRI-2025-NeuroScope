import shutil
import numpy as np
import time
from pathlib import Path
import copy
from collections import defaultdict
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from utils.data.load_data import create_data_loaders, create_indexed_loader
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIM_L1_Loss, SSIMLoss
from utils.model.VarNet import VarNet
from utils.model.FIVarNet import FIVarNet
from utils.learning.Scheduler import CustomWarmupCosineScheduler
from EDA.mask_classifier import classify_and_index


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, scaler):
    """메모리 최적화 구조 그대로 유지"""
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    # gradient accumulation steps (메모리 최적화)
    accumulation_step = args.accumulation_step
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        # GPU 메모리 효율적 이동
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        # Mixed Precision (메모리 최적화)
        with torch.amp.autocast(device_type='cuda'):
            output = model(kspace, mask, is_training=True)
            loss = loss_type(output, target, maximum)
            loss = loss / accumulation_step  # scale loss for accumulation
        
        # AMP scaler (메모리 최적화)
        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_step

        # Gradient accumulation (메모리 최적화)
        if (iter + 1) % accumulation_step == 0 or (iter + 1) == len_loader:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * accumulation_step:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    """메모리 최적화 구조 그대로 유지"""
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    # 메모리 최적화 - gradient 계산 비활성화
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask, is_training=False)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train(args, index_file=None):
    """메모리 최적화 구조 유지하면서 인덱스 지원 추가"""
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # 모델 생성 (기존과 동일)
    if args.model == 'varnet':
        model = VarNet(num_cascades=args.feature_cascades, 
                       chans=args.chans, 
                       sens_chans=args.sens_chans)
    elif args.model == 'fivarnet':
        model = FIVarNet(num_feature_cascades=args.feature_cascades,
                        num_image_cascades=args.image_cascades,
                        use_attn=args.use_attention,
                        chans=args.chans,
                        sens_chans=args.sens_chans)

    model.to(device=device)

    # Loss와 optimizer (기존과 동일)
    loss_type = SSIMLoss().to(device=device)
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        args.lr,
        weight_decay=1e-6
    )

    best_val_loss = 1.
    start_epoch = 0

    # 데이터 로더 생성 - 인덱스 지원 (메모리 구조는 동일)
    if index_file:
        # 인덱스 기반 로딩 (MoE용)
        print("Train with index file")
        train_loader = create_indexed_loader(
            kspace_root="/root/Data/train/kspace",
            image_root="/root/Data/train/image",
            index_file=index_file,
            args=args,
            shuffle=True,
            augmentation=args.data_augmentation
        )
    else:
        # 표준 로딩
        train_loader = create_data_loaders(
            data_path=args.data_path_train, 
            args=args, 
            shuffle=True, 
            augmentation=args.data_augmentation
        )
    
    val_loader = create_data_loaders(
        data_path=args.data_path_val, 
        args=args, 
        shuffle=False, 
        augmentation=False
    )
    
    val_loss_log = np.empty((0, 2))

    # 스케줄러 (기존과 동일)
    scheduler = CustomWarmupCosineScheduler(
        optimizer,
        total_epochs=args.num_epochs,
        warmup1=args.warmup_epochs,
        anneal1=args.anneal1,
        warmup2=args.warmup2,
        anneal2=args.anneal2,
        lr_max1=args.lr,
        lr_min1=args.lr_min1,
        lr_max2=args.lr_max2,
        lr_min2=args.lr_min2
    )

    # Mixed Precision scaler (메모리 최적화)
    scaler = torch.amp.GradScaler()
    
    # 훈련 루프 (메모리 최적화 구조 그대로)
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, scaler)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        # 텐서 변환 (기존과 동일)
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(f'ForwardTime = {time.perf_counter() - start:.4f}s')
        
        scheduler.step(epoch)  # learning rate 업데이트


def train_moe(args):
    """인덱스 기반 MoE 파이프라인 with 클래스별 augmentation"""
    # Step 1: 분류 및 인덱스 파일 생성
    input_folder = "/root/Data/train/kspace"
    output_base = "/root/Data/train/class_indices"
    
    print("Classifying data and creating indices...")
    class_groups, index_files = classify_and_index(input_folder, output_base)
    print("Classification completed!")
    
    # Step 2: 각 클래스별 훈련 (augmentation 포함)
    for class_label, index_file in zip(class_groups.keys(), index_files):
        print(f"\n========== Training for class: {class_label} ==========")
        
        args_class = copy.deepcopy(args)
        args_class.net_name = f"{args.net_name}_{class_label}"
        
        # 클래스별 디렉토리 설정
        args_class.exp_dir = Path(os.path.join(args.result_path, args_class.net_name, 'checkpoints'))
        args_class.val_dir = Path(os.path.join(args.result_path, args_class.net_name, 'reconstructions_val'))
        args_class.main_dir = Path(os.path.join(args.result_path, args_class.net_name, 'main'))
        args_class.val_loss_dir = Path(os.path.join(args.result_path, args_class.net_name))
        
        args_class.exp_dir.mkdir(parents=True, exist_ok=True)
        args_class.val_dir.mkdir(parents=True, exist_ok=True)

        
        # 인덱스 파일로 훈련
        train(args_class, index_file=index_file)
        
        print(f"Training completed for class: {class_label}")
    
    print("\n========== All MoE training completed! ==========")
