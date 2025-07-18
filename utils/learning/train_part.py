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
    """슬라이스 단위 Min-Max 정규화 적용"""
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    accumulation_step = args.accumulation_step
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            if args.model == 'varnet':
                output = model(kspace, mask)
            elif args.model == 'fivarnet':
                output = model(kspace, mask, is_training=True)
            
            # 🔥 슬라이스 단위 Min-Max 정규화
            if iter == 0:
                print(f"🔍 Tensor shapes: output={output.shape}, target={target.shape}")

            
            output_min = output.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            output_max = output.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            target_min = target.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            target_max = target.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        

            output_norm = (output - output_min) / (output_max - output_min + 1e-8)
            target_norm = (target - target_min) / (target_max - target_min + 1e-8)
            
            data_range = torch.tensor(1.0, device=output.device).unsqueeze(0)
            
            # 🔍 스케일 확인 (첫 번째 iter만)
            if iter == 0:
                print(f"\n🔍 Min-Max Normalization:")
                print(f"  Before: output [{output.min():.6f}~{output.max():.6f}], target [{target.min():.6f}~{target.max():.6f}]")
                print(f"  After:  output [{output_norm.min():.6f}~{output_norm.max():.6f}], target [{target_norm.min():.6f}~{target_norm.max():.6f}]")
                print(f"  data_range: {data_range.item():.6f}")
            
            loss = loss_type(output_norm, target_norm, data_range)
            
            # 🔍 Loss 확인 (첫 번째 iter만)
            if iter == 0:
                print(f"  Loss: {loss.item():.6f}")
                print()
            
            loss = loss / accumulation_step

        scaler.scale(loss).backward()
        total_loss += loss.item() * accumulation_step

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
    """검증 실행 (🔥 NaN 문제 해결)"""
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    # 🔥 No Gradient로 메모리 절약
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # 📥 데이터 언패킹
            mask, kspace, target, _, fnames, slices = data
            
            # 🚀 GPU로 이동
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            # 🔥 Target 안전 처리 (NaN 방지)
            target_is_valid = torch.is_tensor(target) and target.numel() > 1
            if target_is_valid:
                target = target.cuda(non_blocking=True)
            
            # 모델 실행
            output = model(kspace, mask, is_training=False)

            # 📊 결과 저장
            for i in range(output.shape[0]):  # 🔥 괄호 추가!
                # Reconstruction 저장
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                
                # 🔥 Target 안전 저장 (문제 해결!)
                if target_is_valid:
                    # 정상적인 target이 있으면 사용
                    targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
                else:
                    # target이 -1이거나 없으면 output 사용 (validation용)
                    targets[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    # 📊 결과 정리
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    
    # 🔥 SSIM Loss 계산
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """모델 저장"""
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
    """메인 훈련 함수 (표준 또는 MoE 지원)"""
    # 🔧 디바이스 설정
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # 🏗️ 모델 생성
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

    # 📊 Loss와 Optimizer
    loss_type = SSIMLoss().to(device=device)
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        args.lr,
        weight_decay=1e-6
    )

    best_val_loss = 1.
    start_epoch = 0

    # 📂 데이터 로더 생성
    if index_file:
        # 🔥 MoE용 인덱스 기반 로딩
        print(f"🔥 MoE Training with index file: {index_file}")
        train_loader = create_indexed_loader(
            kspace_root="/root/Data/train/kspace",
            image_root="/root/Data/train/image",
            index_file=index_file,
            args=args,
            shuffle=True,
            augmentation=args.data_augmentation
        )
    else:
        # 📁 표준 로딩
        print("📁 Standard Training")
        train_loader = create_data_loaders(
            data_path=args.data_path_train, 
            args=args, 
            shuffle=True, 
            augmentation=args.data_augmentation
        )
    
    # 📊 Validation 로더 (항상 표준)
    val_loader = create_data_loaders(
        data_path=args.data_path_val, 
        args=args, 
        shuffle=False, 
        augmentation=False
    )
    
    val_loss_log = np.empty((0, 2))

    # 📈 스케줄러 설정
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

    # 🔥 Mixed Precision Scaler
    scaler = torch.amp.GradScaler()
    
    # 🚀 훈련 루프
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # 📊 훈련 및 검증
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, scaler)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        # 📂 Loss 로그 저장
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"📂 Loss file saved! {file_path}")

        # 🔥 텐서 변환 (메모리 최적화)
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        # 🏆 Best Model 체크
        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        # 💾 모델 저장
        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        # 🏆 New Record 시 결과 저장
        if is_new_best:
            print("🏆 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(f'📊 ForwardTime = {time.perf_counter() - start:.4f}s')
        
        # 📈 Learning Rate 업데이트
        scheduler.step(epoch)


def train_moe(args):
    """🔥 MoE 파이프라인 (클래스별 전문 모델 훈련)"""
    
    # 🔍 기존 분류 파일 확인
    output_base = "/root/Data/train/class_indices"
    existing_files = {
        "acc4-brain": f"{output_base}/acc4-brain.txt",
        "acc4-knee": f"{output_base}/acc4-knee.txt", 
        "acc8-brain": f"{output_base}/acc8-brain.txt",
        "acc8-knee": f"{output_base}/acc8-knee.txt"
    }
    
    # 🔥 기존 파일 존재 여부 확인
    all_files_exist = all(os.path.exists(file) for file in existing_files.values())
    
    if all_files_exist:
        print("🔥 Found existing classification files! Skipping classification step...")
        class_groups = existing_files
        index_files = list(existing_files.values())
    else:
        print("🔍 Classification files not found. Running classification...")
        # 🔍 Step 1: 데이터 분류 및 인덱스 생성
        input_folder = "/root/Data/train/kspace"
        class_groups, index_files = classify_and_index(input_folder, output_base)
        print("✅ Classification completed!")
    
    # 🚀 Step 2: 각 클래스별 전문 모델 훈련
    for class_label, index_file in zip(class_groups.keys(), index_files):
        print(f"\n🔥 ========== Training for class: {class_label} ==========")
        
        # 📋 클래스별 args 복사
        args_class = copy.deepcopy(args)
        args_class.net_name = f"{args.net_name}_{class_label}"
        
        # 📂 클래스별 디렉토리 설정
        args_class.exp_dir = Path(os.path.join(args.result_path, args_class.net_name, 'checkpoints'))
        args_class.val_dir = Path(os.path.join(args.result_path, args_class.net_name, 'reconstructions_val'))
        args_class.main_dir = Path(os.path.join(args.result_path, args_class.net_name, 'main'))
        args_class.val_loss_dir = Path(os.path.join(args.result_path, args_class.net_name))
        
        # 📁 디렉토리 생성
        args_class.exp_dir.mkdir(parents=True, exist_ok=True)
        args_class.val_dir.mkdir(parents=True, exist_ok=True)

        # 🚀 클래스별 전문 모델 훈련
        train(args_class, index_file=index_file)
        
        print(f"✅ Training completed for class: {class_label}")
    
    print("\n🎉 ========== All MoE training completed! ==========")
