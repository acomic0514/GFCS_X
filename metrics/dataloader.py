

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import numpy as np
import pandas as pd
import re  # 新增：正則化模組用於語意標註
from torchvision.transforms import functional as F

# 自動找到 `GFCS_X` 目錄
GFCS_X_ROOT = Path(__file__).resolve().parent.parent  # `metrics` 的上級目錄
DATASET_ROOT = GFCS_X_ROOT / "data"  # `GFCS_X/data/`


class RainDataset(Dataset):
    def __init__(self, mode='train', dataset_name=None, 
                 transform=None, inference=False, return_size=False):
        """
        mode: 'train' 或 'test'
        dataset_name: 數據集名稱，如 'Rain13K' 或 'Rain100L'
        transform: 圖片增強 / 標準化
        test: 是否為測試模式（允許 target 為 None）
        """
        assert mode in ['train', 'test'], "mode 參數必須是 'train' 或 'test'"
        self.input_dir = DATASET_ROOT / mode / dataset_name / 'input'
        self.target_dir = DATASET_ROOT / mode / dataset_name / 'target' if not inference else None
        self.transform = transform
        self.inference = inference
        self.return_size = return_size  # 是否返回原始尺寸

        # 確保目錄存在
        if not self.input_dir.exists():
            raise FileNotFoundError(f"❌ 找不到目錄 {self.input_dir}，請檢查數據集！")

        if not inference and not self.target_dir.exists():
            raise FileNotFoundError(f"❌ 找不到目錄 {self.target_dir}，請檢查數據集！")

        # 讀取圖像
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.png', '.jpg'))])

        if not inference:
            self.target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith(('.png', '.jpg'))])
            assert len(self.input_files) == len(self.target_files), "❌ input 和 target 數量不匹配！"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_filename = self.input_files[idx]
        input_path = self.input_dir / self.input_files[idx]
        input_img = Image.open(input_path).convert("RGB")
        # orig_size = input_img.size  # 保存原始尺寸 (W, H)

        if not self.inference:
            target_filename = self.target_files[idx]
            target_path = self.target_dir / self.target_files[idx]
            target_img = Image.open(target_path).convert("RGB")
        else:
            target_img = None
            target_filename = None

        # 轉換圖像
        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        
        if not self.inference:
            sample = {
            "input": input_img,
            "target": target_img,
            "input_name": input_filename,
            "target_name": target_filename,
            # "orig_size": orig_size if self.return_size else None
            }
        else:
            sample = {
            "input": input_img,
            "input_name": input_filename,
            # "orig_size": orig_size if self.return_size else None
            }

        return sample



# 自定義影像對處理類別
# 這個類別用於對一對影像進行同步處理
class PairedCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_img, target_img):
        for t in self.transforms:
            input_img, target_img = t(input_img, target_img)
        return input_img, target_img
    
# 同步裁切
class PairedRandomResizedCrop:
    def __init__(self, size, scale=(0.8, 1.0), ratio=(1.0, 1.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, input_img, target_img):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            input_img, scale=self.scale, ratio=self.ratio
        )
        input_img = F.resized_crop(input_img, i, j, h, w, self.size)
        target_img = F.resized_crop(target_img, i, j, h, w, self.size)
        return input_img, target_img
    
# 同步翻轉
class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_img, target_img):
        if random.random() < self.p:
            return F.hflip(input_img), F.hflip(target_img)
        return input_img, target_img
    
# 同步旋轉
class PairedRandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, input_img, target_img):
        angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
        return (
            F.rotate(input_img, angle),
            F.rotate(target_img, angle)
        )
def to_tensor_pair(input_img, target_img):
    input_tensor = transforms.ToTensor()(input_img)
    if target_img is not None:
        target_tensor = transforms.ToTensor()(target_img)
    else:
        target_tensor = None
    return input_tensor, target_tensor
        
def get_transform(train=True):
    if train:
        return PairedCompose([
            PairedRandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            # PairedRandomHorizontalFlip(p=0.5),
            # PairedRandomRotation(degrees=10),
            # 加入轉 tensor：為保證一致，最後一併轉
            to_tensor_pair
        ])
    else:
        return to_tensor_pair
"""
def get_transform(train=True):
    if train:
        return PairedCompose([
            PairedRandomResizedCrop(size=(128, 128), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            # PairedRandomHorizontalFlip(p=0.5),
            # PairedRandomRotation(degrees=10),
            # 加入轉 tensor：為保證一致，最後一併轉
            lambda x, y: (transforms.ToTensor()(x), transforms.ToTensor()(y))
        ])
    else:
        return lambda x, y: (transforms.ToTensor()(x), transforms.ToTensor()(y))
"""
    
# def get_transform(mean, std, train=True): 
# def get_transform(train=True): 
#     if train:
#         return transforms.Compose([
#             # transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
#             # transforms.RandomRotation(10),  # 隨機旋轉 ±10 度
#             # transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # 隨機裁剪
#             transforms.RandomResizedCrop(128),  # 隨機裁剪
#             transforms.ToTensor(),  # 轉換為 Tensor
#             #transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # 標準化
#         ])
#     else:
#         return transforms.Compose([
#             #transforms.Resize((256, 256)),  # 測試時直接縮放到 256x256
#             transforms.ToTensor(),
#             #transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # 標準化
#         ])

###########################################################################
# Save trace data to CSV
def save_trace_dict_to_csv(trace_dict, prefix_dir="trace_output"):
    os.makedirs(prefix_dir, exist_ok=True)
    for name, value in trace_dict.items():
        if isinstance(value, np.ndarray):
            print(f"📁 正在儲存: {name}, shape: {value.shape}, ndim: {value.ndim}")  
            # 🔍 加入語意化命名提示
            semantic_name = name
            semantic_name = semantic_name.replace("inconv1", "input_projection")
            semantic_name = semantic_name.replace("outconv1", "output_projection")
            semantic_name = semantic_name.replace("convD1", "ddrb_branch1")
            semantic_name = semantic_name.replace("convD2", "ddrb_branch2")
            semantic_name = semantic_name.replace("convD3", "ddrb_branch3")
            
            if value.ndim == 2:
                df = pd.DataFrame(value)
                df.to_csv(os.path.join(prefix_dir, f"{name}.csv"), index=False)
            elif value.ndim == 3:
                for c in range(value.shape[0]):
                    df = pd.DataFrame(value[c])
                    df.to_csv(os.path.join(prefix_dir, f"{name}_ch{c}.csv"), index=False)
            elif value.ndim == 4:
                if "output" in name:
                    with open(os.path.join(prefix_dir, f"{semantic_name}.csv"), "w") as f:
                        C = value.shape[1]
                        for ch in range(C):
                            f.write(f"# {semantic_name}_ch{ch}\n")
                            np.savetxt(f, value[0, ch], delimiter=",", fmt="%.6f")
                            f.write("\n")
                elif "weight" in name:
                    for out_c in range(value.shape[0]):
                        with open(os.path.join(prefix_dir, f"{semantic_name}_out{out_c}.csv"), "w") as f:
                            for in_c in range(value.shape[1]):
                                f.write(f"# in_channel {in_c}\n")
                                np.savetxt(f, value[out_c, in_c], delimiter=",", fmt="%.6f")
                                f.write("\n")
                                f.write("# ---------------------------------------------\n")
                else:
                    np.save(os.path.join(prefix_dir, f"{semantic_name}.npy"), value)


# # 計算 mean/std 的獨立函數
# def compute_mean_std(mode='train', dataset_name='Rain13K', device='cpu'):
#     """
#     計算指定資料夾內所有圖片的 mean 和 std
#     """
#     input_dir = DATASET_ROOT / mode / dataset_name / 'input'  # 自動獲取 input 圖像路徑
#     target_dir = DATASET_ROOT / mode / dataset_name / 'target'  # 自動獲取 target 圖像路徑

#     if not input_dir.exists() or not target_dir.exists():
#         raise FileNotFoundError(f"❌ 找不到數據集目錄: {input_dir}")
    
#     temp_transform = transforms.ToTensor()  # 強制轉換為 Tensor
    
#     device = torch.device(device) # 設置設備

#     input_mean = torch.zeros(3, device=device)
#     input_std = torch.zeros(3, device=device)
#     target_mean = torch.zeros(3, device=device)
#     target_std = torch.zeros(3, device=device)
#     n_samples = 0

#     print(f"📊 正在計算 {mode}/{dataset_name} 數據集的 mean 和 std...")

#     input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])
#     target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg'))])
#     if len(input_files) == 0 or len(target_files) == 0:
#         raise ValueError("❌ 沒有找到圖像文件，請檢查數據集路徑！")
    
#     for input_file, target_file in zip(input_files, target_files):
#         input_img = Image.open(input_dir / input_file).convert("RGB")
#         target_img = Image.open(target_dir / target_file).convert("RGB")

#         input_img = temp_transform(input_img).to(device)   # 轉換為 Tensor
#         target_img = temp_transform(target_img).to(device) 

#         input_mean += input_img.mean(dim=(1, 2))
#         input_std += input_img.std(dim=(1, 2))

#         target_mean += target_img.mean(dim=(1, 2))
#         target_std += target_img.std(dim=(1, 2))

#         n_samples += 1

#     input_mean /= n_samples
#     input_std /= n_samples
#     target_mean /= n_samples
#     target_std /= n_samples
    
    
#     print(f"✅ 計算完成：")
#     print(f"📊 訓練數據集的 input mean: {input_mean.tolist()}, std: {input_std.tolist()}")
#     print(f"📊 訓練數據集的 target mean: {target_mean.tolist()}, std: {target_std.tolist()}")

#     return input_mean, input_std, target_mean, target_std