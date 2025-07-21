
import shutil
import numpy as np
import time
from pathlib import Path
import copy
from collections import defaultdict
import os
from tqdm import tqdm

import torch
from torch import Tensor
from torch import optim
import torch.nn as nn

from utils.data.load_data import create_data_loaders
from utils.common.utils import ssim_loss
from utils.common.loss_function import SSIM_L1_Loss, SSIMLoss
from utils.model.VarNet import VarNet
from utils.model.FIVarNet import FIVarNet
from utils.learning.Scheduler import ConstantScheduler, CosineScheduler, WarmupCosineScheduler, DoubleWarmupCosineScheduler
from utils.learning.mask_classifier import classify_and_index


class FastMRI():
    def __init__(self, args):
        self.args = args

        self.device = self._get_device(args)
        self.model = self._build_model(args).to(self.device)


    def _get_device(self, args):
        device = torch.device(f'cuda:{args.GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')    
        torch.cuda.set_device(device)
        print(f'Current device: {device}')
        return device
    

    def _build_model(self, args):
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
        else:
            raise NotImplementedError(f"Invalid model: {args.model}")
        return model
    

    def _select_optimizer(self):
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), 
                                   lr=self.args.lr)
        elif self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=1e-6)
        else:
            raise NotImplementedError(f"Invalid optimizer type: {self.args.optimizer}")
        return optimizer
    

    def _select_criterion(self):
        if self.args.criterion == 'SSIM':
            criterion = SSIMLoss()
        elif self.args.criterion == 'SSIM_L1':
            criterion = SSIM_L1_Loss()  
        else:
            raise NotImplementedError(f"Invalid loss type: {self.args.criterion}")
        return criterion
    

    def _select_scheduler(self, optimizer):
        if self.args.scheduler == 'cosine':
            scheduler = CosineScheduler(optimizer, 
                                        total_epochs=self.args.num_epochs, 
                                        lr_max=self.args.lr,
                                        lr_min=self.args.lr_min1)
        elif self.args.scheduler == 'constant':
            scheduler = ConstantScheduler(optimizer, lr=self.args.lr)
        elif self.args.scheduler == 'warmup_cosine':
            scheduler = WarmupCosineScheduler(optimizer, 
                                             total_epochs=self.args.num_epochs, 
                                             warmup_epochs=self.args.warmup1,
                                             lr_max=self.args.lr)
        elif self.args.scheduler == 'double_warmup_cosine':
            scheduler = DoubleWarmupCosineScheduler(
                optimizer,
                total_epochs=self.args.num_epochs,
                warmup1=self.args.warmup1,
                anneal1=self.args.anneal1,
                warmup2=self.args.warmup2,
                anneal2=self.args.anneal2,
                lr_max1=self.args.lr,
                lr_min1=self.args.lr_min1,
                lr_max2=self.args.lr_max2,
                lr_min2=self.args.lr_min2
            )
            
        else:
            raise NotImplementedError(f"Invalid scheduler type: {self.args.scheduler}")
        
        return scheduler


    def _select_scaler(self):
        return torch.amp.GradScaler()
    

    def count_parameters(self):
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        print(f"Number of Trainable Params: {sum(params)}")
    

    def print_model(self):
        print(f"Model: {self.args.net_name}")
        print(str(self.model))


    def save_model(self, model_name, exp_dir, epoch, val_loss, optimizer, is_best=False):
        save_name = f"e{epoch}_loss{val_loss:.4f}.pt"
        torch.save(
            {
                'name': model_name,
                'model': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            f=exp_dir / save_name
        )
        if is_best:
            shutil.copyfile(exp_dir / save_name,
                            exp_dir / "best_model.pt")


    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'])


    def validate(self, val_loader):
        self.model.eval()
        reconstructions = defaultdict(dict)
        targets = defaultdict(dict)
        start = time.perf_counter()

        with torch.no_grad():
            for _, data in enumerate(val_loader):
                mask, kspace, target, _, fnames, slices = data
            
                kspace = kspace.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                
                target_is_valid = torch.is_tensor(target) and target.numel() > 1
                if target_is_valid:
                    target = target.cuda(non_blocking=True)
                
                output = self.model(kspace, mask, is_training=False)

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    
                    if target_is_valid:
                        targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()
                    else:
                        targets[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

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
        return metric_loss, num_subjects, reconstructions, targets, time.perf_counter() - start
    

    def train_epoch(self, epoch, train_loader, optimizer, criterion, scaler, fold):
        self.model.train()
        start_epoch = start_iter = time.perf_counter()
        len_loader = len(train_loader)
        total_loss = 0.

        accumulation_step = self.args.accumulation_step
        optimizer.zero_grad()

        for iter, data in enumerate(train_loader):
            mask, kspace, target, maximum, _, _ = data
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            output = self.model(kspace, mask, is_training=True)
            data_range = maximum

            loss = criterion(output, target, data_range)
            loss = loss / accumulation_step

            loss.backward()
            total_loss += loss.item() * accumulation_step

            if (iter + 1) % accumulation_step == 0 or (iter + 1) == len_loader:
                optimizer.step()
                optimizer.zero_grad()

            if (iter + 1) % self.args.report_interval == 0:
                log_template = f'Epoch = [{epoch + 1:3d}/{self.args.num_epochs:3d}]  ' + \
                    (f'Fold = [{fold + 1:2d}/{self.args.num_folds:2d}]  ' if self.args.k_fold else '') + \
                    f'Iter = [{iter + 1:4d}/{len_loader:4d}]  ' + \
                    f'Loss = {loss.item() * accumulation_step:.5f}  ' + \
                    f'Time = {time.perf_counter() - start_iter:.4f}s'
                print(log_template)
                start_iter = time.perf_counter()

        total_loss = total_loss / len_loader
        return total_loss, time.perf_counter() - start_epoch


    def train_single_class(self, class_label, index_file=None, exp_dir="./checkpoint", val_loss_dir="./"):
        print(f"\n=============== Training for class: {class_label} ===============")

        best_val_loss = 1.
        start_epoch = 0
        val_loss_log = np.empty((0, 2))

        criterion = self._select_criterion().to(self.device)
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer=optimizer)
        scaler = self._select_scaler()

        model_name = f"{self.args.net_name}_{class_label}"
        fold_num = self.args.num_folds if self.args.k_fold else 1
        for epoch in range(start_epoch, self.args.num_epochs):
            print(f'Epoch #{epoch:3d} =============== {model_name} ===============')
            scheduler.adjust_lr(epoch)

            # K-Fold cross-validation
            for val_fold in range(fold_num):
                print(f"Fold {val_fold + 1}/{fold_num} for class {class_label}")
                train_loader, val_loader = next(create_data_loaders(
                    train_data_path=self.args.data_path_train,
                    val_data_path=self.args.data_path_val,
                    args=self.args,
                    index_file=index_file,
                    shuffle=True,
                    isforward=False,
                    augmentation=self.args.data_augmentation
                ))
        
                train_loss, train_time = self.train_epoch(epoch=epoch,
                                                         train_loader=train_loader, 
                                                         optimizer=optimizer, 
                                                         criterion=criterion, 
                                                         scaler=scaler,
                                                         fold=val_fold)
                val_loss, num_subjects, _, _, val_time = self.validate(val_loader)
                
                val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
                val_log_path = os.path.join(val_loss_dir, "val_loss_log")
                np.save(val_log_path, val_loss_log)

                train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
                val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
                val_loss = val_loss / num_subjects
                
                is_new_best = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)

                self.save_model(model_name=model_name, exp_dir=exp_dir, epoch=epoch, val_loss=val_loss, optimizer=optimizer, is_best=is_new_best)
                print(
                    f'Epoch = [{epoch + 1:3d}/{self.args.num_epochs:3d}]   '
                    f'TrainLoss = {train_loss:.4g}   '
                    f'ValLoss = {val_loss:.4g}   '
                    f'TrainTime = {train_time:.4f}s   '
                    f'ValTime = {val_time:.4f}s   '
                )
    

    def class_split(self):
        output_base = self.args.class_split_path
        class_groups = {
            "acc4-brain": f"{output_base}/acc4-brain.txt",
            "acc4-knee": f"{output_base}/acc4-knee.txt", 
            "acc8-brain": f"{output_base}/acc8-brain.txt",
            "acc8-knee": f"{output_base}/acc8-knee.txt"
        }
        
        need_classification = not all(os.path.exists(file) for file in class_groups.values())
        
        if need_classification:
            print("Classification files not found. Running classification...")
            train_dir = self.args.data_path_train / self.args.input_key
            val_dir = self.args.data_path_val / self.args.input_key
            _, _ = classify_and_index(train_dir, val_dir, output_base)
            print("Classification completed!")
        else:
            print("Found existing classification files! Skipping classification step...")
        
        return class_groups
    

    @staticmethod
    def set_class_directory_path(net_name, class_label, result_path):
        
        model_name = f"{net_name}_{class_label}"
        
        exp_dir = Path(os.path.join(result_path, model_name, 'checkpoints'))
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        val_loss_dir = Path(os.path.join(result_path, model_name))
        
        return exp_dir, val_loss_dir


    def train(self):
        class_groups = self.class_split() if self.args.use_moe else {"all": None}

        for class_label, index_file in class_groups.items():
            
            exp_dir, val_loss_dir = self.set_class_directory_path(net_name=self.args.net_name, 
                                                                  class_label=class_label,
                                                                  result_path=self.args.result_path)
            self.train_single_class(class_label=class_label, 
                                    index_file=index_file,
                                    exp_dir=exp_dir,
                                    val_loss_dir=val_loss_dir)
            
            print(f"Training completed for class: {class_label}")
        
        print("\n========== Training completed! ==========")