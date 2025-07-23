
import numpy as np
import time
from collections import defaultdict
import os
import cv2 
import h5py
import glob

import torch
import torch.nn.functional as F

from utils.data.load_data import create_eval_loaders
from utils.model.VarNet import VarNet
from utils.model.FIVarNet import FIVarNet
from utils.learning.mask_classifier import MRIClassifier
from utils.common.utils import save_reconstructions
from utils.common.loss_function import SSIMLoss


class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)
            
    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))
            
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return S.mean()
    

def forward(args, leaderboard_data_path=None, your_data_path=None):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    leaderboard_data = glob.glob(os.path.join(leaderboard_data_path, '*.h5'))
    if len(leaderboard_data) != 58:
        raise  NotImplementedError('Leaderboard Data Size Should Be 58')

    your_data = glob.glob(os.path.join(your_data_path, '*.h5'))
    if len(your_data) != 58:
        raise  NotImplementedError('Your Data Size Should Be 58')           
    
    ssim_total = 0
    idx = 0
    ssim_calculator = SSIM().to(device=device)
    with torch.no_grad():
        for part in ['brain_test', 'knee_test']:
            for i_subject in range(29):
                l_fname = os.path.join(leaderboard_data_path, part + str(i_subject+1) + '.h5')
                y_fname = os.path.join(your_data_path, part + str(i_subject+1) + '.h5')
                with h5py.File(l_fname, "r") as hf:
                    num_slices = hf['image_label'].shape[0]
                for i_slice in range(num_slices):
                    with h5py.File(l_fname, "r") as hf:
                        target = hf['image_label'][i_slice]
                        mask = np.zeros(target.shape)
                        if part == 'knee_test':
                            mask[target>2e-5] = 1
                        elif part == 'brain_test':
                            mask[target>5e-5] = 1
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                        mask = cv2.dilate(mask, kernel, iterations=15)
                        mask = cv2.erode(mask, kernel, iterations=14)
                        
                        target = torch.from_numpy(target).to(device=device)
                        mask = (torch.from_numpy(mask).to(device=device)).type(torch.float)

                        maximum = hf.attrs['max']
                        
                    with h5py.File(y_fname, "r") as hf:
                        recon = hf[args.output_key][i_slice]
                        recon = torch.from_numpy(recon).to(device=device)
                        
                    ssim_total += ssim_calculator(recon*mask, target*mask, maximum).cpu().numpy()
                    idx += 1
            
    return ssim_total/idx

class EvalMRI:
    """Class for evaluating MRI models"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._get_device(args)
        self.model = self._build_model(args).to(self.device)
        self.model.eval()
        self.mri_classifier = self._select_MRI_classifier()

        self.current_model = None
        self.ckpt_path = {
            'brain-acc4': args.brain_acc4_ckpt,
            'brain-acc8': args.brain_acc8_ckpt,
            'knee-acc4': args.knee_acc4_ckpt,
            'knee-acc8': args.knee_acc8_ckpt
        }


    def _get_device(self, args):
        device = torch.device(f'cuda:{args.GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')    
        torch.cuda.set_device(device)
        print(f'Current device: {device}')
        return device
      

    def _build_model(self, args):
        if args.model == 'VarNet':
            return VarNet(args)
        elif args.model == 'FIVarNet':
            return FIVarNet(args)
        else:
            raise NotImplementedError(f"Model {args.model} is not implemented.")
        

    def _select_MRI_classifier(self):
        return MRIClassifier()
    

    def load_model(self, model_type):
        assert model_type in self.ckpt_path.keys(), f"Invalid model type: {model_type}"
        checkpoint_path = self.ckpt_path[model_type]
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model'])
        self.current_model = model_type
    

    def reconstruct(self, data_path=None, forward_dir=None):
        reconstructions = defaultdict(dict)
        data_loader = create_eval_loaders(data_path=data_path,
                                          args=self.args,
                                          isforward=True)

        with torch.no_grad():
            for (mask, kspace, _, _, fnames, slices) in data_loader:
                kspace = kspace.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                classification = self.mri_classifier.classify(kspace, mask)
                anatomy = classification['anatomy']
                acceleration = classification['acceleration']
                
                model_type = f"{anatomy}-{acceleration}"
                if self.current_model != model_type:
                    self.load_model(model_type)
                    print(f"Loaded model: {model_type}")

                output = self.model(kspace, mask, is_training=False)

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
        
        save_reconstructions(reconstructions, forward_dir, targets=None, inputs=None)
        print(f"Reconstructions saved to {forward_dir}")


    def run_reconstruction(self):
        """Run the reconstruction process for leaderboard evaluation"""
        if not self.args.forward_dir.exists():
            self.args.forward_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # acc4
        acc4_data_path = self.args.leaderboard_path / "acc4"
        acc4_save_path = self.args.forward_dir / "acc4"
        self.reconstruct(data_path=acc4_data_path,
                         forward_dir=acc4_save_path)

        # acc8
        acc8_data_path = self.args.leaderboard_path / "acc8"
        acc8_save_path = self.args.forward_dir / "acc8"
        self.reconstruct(data_path=acc8_data_path,
                         forward_dir=acc8_save_path)

        reconstructions_time = time.time() - start_time
        
        print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

        reconstruction_success = reconstructions_time < 3600
        print('Reconstruction Success!') if reconstruction_success else print('Reconstruction Fail!')


    def leaderboard_eval(self):
        """Evaluate the model on the leaderboard dataset"""
        
        # acc4
        acc4_data_path = self.args.leaderboard_path / "acc4"
        acc4_save_path = self.args.forward_dir / "acc4"
        SSIM_acc4 = forward(self.args,
                            leaderboard_data_path=acc4_data_path,
                            your_data_path=acc4_save_path)
        # acc8
        acc8_data_path = self.args.leaderboard_path / "acc8"
        acc8_save_path = self.args.forward_dir / "acc8"
        SSIM_acc8 = forward(self.args,
                            leaderboard_data_path=acc8_data_path,
                            your_data_path=acc8_save_path)

        print("Leaderboard SSIM : {:.4f}".format((SSIM_acc4 + SSIM_acc8) / 2))
        print("="*10 + " Details " + "="*10)
        print("Leaderboard SSIM (acc4): {:.4f}".format(SSIM_acc4))
        print("Leaderboard SSIM (acc8): {:.4f}".format(SSIM_acc8))
    
    def evaluate(self):
        self.run_reconstruction()
        self.leaderboard_eval()