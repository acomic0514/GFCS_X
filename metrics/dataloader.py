#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


# In[ ]:


# ✅ 自動找到 `GFCS_X` 目錄
GFCS_X_ROOT = Path(__file__).resolve().parent.parent  # `metrics` 的上級目錄
DATASET_ROOT = GFCS_X_ROOT / "data"  # `GFCS_X/data/`


# In[5]:


class RainDataset(Dataset):
    def __init__(self, mode='train', dataset_name='Rain13K', transform=None, test=False):
        """
        mode: 'train' 或 'test'
        dataset_name: 數據集名稱，如 'Rain13K' 或 'Rain100L'
        transform: 圖片增強 / 標準化
        test: 是否為測試模式（允許 target 為 None）
        """
        assert mode in ['train', 'test'], "mode 參數必須是 'train' 或 'test'"

        self.input_dir = DATASET_ROOT / mode / dataset_name / 'input'
        self.target_dir = DATASET_ROOT / mode / dataset_name / 'target' if not test else None
        self.transform = transform
        self.test = test

        # ✅ 確保目錄存在
        if not self.input_dir.exists():
            raise FileNotFoundError(f"❌ 找不到目錄 {self.input_dir}，請檢查數據集！")

        if not test and not self.target_dir.exists():
            raise FileNotFoundError(f"❌ 找不到目錄 {self.target_dir}，請檢查數據集！")

        # ✅ 讀取圖像
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(('.png', '.jpg'))])

        if not test:
            self.target_files = sorted([f for f in os.listdir(self.target_dir) if f.endswith(('.png', '.jpg'))])
            assert len(self.input_files) == len(self.target_files), "❌ input 和 target 數量不匹配！"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_dir / self.input_files[idx]
        input_img = Image.open(input_path).convert("RGB")

        if not self.test:
            target_path = self.target_dir / self.target_files[idx]
            target_img = Image.open(target_path).convert("RGB")
        else:
            target_img = None

        # 轉換圖像
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img) if target_img else None

        return (input_img, target_img) if target_img is not None else input_img



# In[ ]:




