import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################################################
## L1 Loss
class L1Loss(nn.Module):
    """L1 Loss (Mean Absolute Error) with optional weight.
    Args:
        loss_weight (float): L1 Loss 的加權係數，默認為 1.0。
        reduction (str): 損失的縮減方式，可選 'none' | 'mean' | 'sum'，默認 'mean'。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], f'Invalid reduction mode: {reduction}'
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
        """
        Args:
            pred (Tensor): 預測圖像，形狀為 (N, C, H, W)。
            target (Tensor): 真實圖像，形狀為 (N, C, H, W)。
            weight (Tensor, optional): 權重張量，形狀為 (N, C, H, W)，默認 None。

        Returns:
            Tensor: 計算後的 L1 Loss。
        """
        # pred, target = pred.half(), target.half() # 確保輸入是 float16
        loss = F.l1_loss(pred, target, reduction='none')  # 計算逐像素 L1 Loss
        
        # 如果提供了權重，則進行加權
        if weight is not None:
            assert weight.shape == loss.shape, "Weight tensor must have the same shape as loss tensor."
            loss = loss * weight# 確保權重也轉為 float16

        # 根據 reduction 進行縮減
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        return self.loss_weight * loss  # 'none' 模式下直接返回逐像素 Loss
