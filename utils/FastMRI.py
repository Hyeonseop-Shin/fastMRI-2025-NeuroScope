
import shutil
import numpy as np
from pathlib import Path
import os

import torch
from torch import Tensor
from torch import optim

from utils.data.load_data import create_data_loaders
from utils.common.loss_function import SSIM_L1_Loss, SSIMLoss
from utils.model.VarNet import VarNet
from utils.model.FIVarNet import FIVarNet

from utils.learning.Scheduler import ConstantScheduler, CosineScheduler, WarmupCosineScheduler, DoubleWarmupCosineScheduler
from utils.learning.mask_classifier import classify_and_index, MRIClassifier
from utils.learning.train_part import train_epoch, validate


class FastMRI:
    def __init__(self, args):
        self.args = args

        self.device = self._get_device(args)
        self.model = self._build_model(args).to(self.device)
        self.mri_classifier = self._select_MRI_classifier()


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
    
    def _select_MRI_classifier(self):
        return MRIClassifier()
    

    def _select_scheduler(self, optimizer):
        eff_epochs = self.args.num_epochs if not self.args.k_fold else self.args.num_epochs * self.args.num_folds

        if self.args.scheduler == 'cosine':
            scheduler = CosineScheduler(optimizer, 
                                        total_epochs=eff_epochs, 
                                        lr_max=self.args.lr,
                                        lr_min=self.args.lr_min1)
        elif self.args.scheduler == 'constant':
            scheduler = ConstantScheduler(optimizer, lr=self.args.lr)
        elif self.args.scheduler == 'warmup_cosine':
            scheduler = WarmupCosineScheduler(optimizer,
                                             total_epochs=eff_epochs,
                                             warmup_epochs=self.args.warmup1,
                                             lr_max=self.args.lr)
        elif self.args.scheduler == 'double_warmup_cosine':
            scheduler = DoubleWarmupCosineScheduler(
                optimizer,
                total_epochs=eff_epochs,
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


    def reset_model(self, retrain=False, class_label="all"):

        if retrain:
            ckpt_file = f"{self.args.net_name}_{class_label}/checkpoints/best_model.pt"
            ckpt_path = self.args.result_path / ckpt_file
            self.load_model(ckpt_path=ckpt_path)

        else:   # from_scratch
            if hasattr(self, "model"):
                del self.model
                torch.cuda.empty_cache()

            self.model = self._build_model(self.args).to(self.device)
            self.optimizer = self._select_optimizer()
            self.scheduler = self._select_scheduler(self.optimizer)


    def save_model(self, model_name, exp_dir, epoch, val_loss, optimizer, fold, is_best=False):
        save_name = f"epoch{epoch}_fold{fold}_loss{val_loss:.4f}.pt"
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


    def log_val_loss(self, val_loss_log: np.ndarray, epoch: int, val_loss: float):
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        os.makedirs(self.args.val_loss_dir, exist_ok=True)
        
        val_log_path = os.path.join(self.args.val_loss_dir, "val_loss_log.npy")
        np.save(val_log_path, val_loss_log)
        
        return val_loss_log


    def train_single_class(self, class_label, index_file=None, exp_dir="./checkpoint", val_loss_dir="./"):
        print(f"\n=============== Training for class: {class_label} ===============")

        best_val_loss = 1.
        val_loss_log = np.empty((0, 2))
        start_epoch = self.args.start_epoch
        last_epoch = start_epoch + self.args.num_epochs

        criterion = self._select_criterion().to(self.device)
        optimizer = self._select_optimizer()
        scheduler = self._select_scheduler(optimizer=optimizer)
        scaler = self._select_scaler()

        model_name = f"{self.args.net_name}_{class_label}"
        fold_num = self.args.num_folds if self.args.k_fold else 1
        for epoch in range(start_epoch, last_epoch):
            print(f'Epoch #{epoch+1:3d} =============== {model_name} ===============')

            # K-Fold cross-validation
            for val_fold in range(fold_num):
                scheduler.adjust_lr(epoch*fold_num + val_fold)

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
        
                train_loss, train_time = train_epoch(model=self.model,
                                                     epoch=epoch,
                                                     train_loader=train_loader, 
                                                     optimizer=optimizer, 
                                                     criterion=criterion, 
                                                     scaler=scaler,
                                                     fold=val_fold,
                                                     args=self.args)
                train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
                
                val_loss, num_subjects, _, _, val_time = validate(self.model, val_loader)
                val_loss_log = self.log_val_loss(val_loss_log, epoch, val_loss)
                val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
                val_loss = val_loss / num_subjects
                
                is_new_best = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)

                self.save_model(model_name=model_name, exp_dir=exp_dir, epoch=epoch, val_loss=val_loss, optimizer=optimizer, fold=val_fold, is_best=is_new_best)
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
            self.reset_model(retrain=self.args.retrain, class_label=class_label)  # Reset model parameters before training each class
            self.train_single_class(class_label=class_label, 
                                    index_file=index_file,
                                    exp_dir=exp_dir,
                                    val_loss_dir=val_loss_dir)
            
            print(f"Training completed for class: {class_label}")
        
        print("\n========== Training completed! ==========")