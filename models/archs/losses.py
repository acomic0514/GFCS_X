import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
import kornia.filters as KF
from math import exp
from torch.autograd import Variable

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

##########################################################################
# SSIM Loss
class SSIMLoss(torch.nn.Module):

    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        """
        Usage:
        ssim_loss = SSIMLoss()
        loss = ssim_loss(pred, target)
        """
        return 1 - ssim(img1, img2, data_range=1.0)

##########################################################################
# Edge Loss
class EdgeLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super(EdgeLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, img1, img2):
        img1_edges = KF.laplacian(img1, kernel_size=3)
        img2_edges = KF.laplacian(img2, kernel_size=3)
        edge_diff = torch.sqrt((img1_edges - img2_edges).pow(2) + self.epsilon ** 2)
        return edge_diff.mean()

##########################################################################
# **SSIM Loss (基於論文版本)**
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

#
def check_nan_inf(x, comment=""):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("⚠️ 發現 NaN 或 Inf，檢查哪張圖片有問題...")
        
        # 找出包含 NaN 或 Inf 的圖片索引
        problematic_indices = []
        for i in range(x.shape[0]):
            if torch.isnan(x[i]).any() or torch.isinf(x[i]).any():
                problematic_indices.append(i)
        print(f"❌ comment: {comment}")
        print(f"❌ 有問題的圖片索引：{problematic_indices}")
        # print(f"❌ 有問題得圖片數值為：{x[problematic_indices]}")
        print(f"❌ 有問題的圖片最大數值為：{x[problematic_indices].max().item()}")
        print(f"❌ 有問題的圖片最小數值為：{x[problematic_indices].min().item()}")
        
        return True
#

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # check_nan_inf(img1, "img1")
    # check_nan_inf(img2, "img2")
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    # check_nan_inf(mu1, "mu1")
    
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # check_nan_inf(mu2, "mu2")

    mu1_sq = mu1.pow(2)
    # check_nan_inf(mu1_sq, "mu1_sq")
    
    mu2_sq = mu2.pow(2)
    # check_nan_inf(mu2_sq, "mu2_sq")
    
    mu1_mu2 = mu1 * mu2
    # check_nan_inf(mu1_mu2, "mu1_mu2")

    sigma1_sq_conv = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
    # check_nan_inf(sigma1_sq_conv, "sigma1_sq_conv")
    sigma1_sq = sigma1_sq_conv - mu1_sq
    # check_nan_inf(sigma1_sq, "sigma1_sq")
    
    sigma2_sq_conv = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
    # check_nan_inf(sigma2_sq_conv, "sigma2_sq_conv")
    sigma2_sq = sigma2_sq_conv - mu2_sq
    # check_nan_inf(sigma1_sq, "sigma1_sq")
    
    sigma12_conv = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
    # check_nan_inf(sigma12_conv, "sigma12_conv")
    sigma12 = sigma12_conv - mu1_mu2
    # check_nan_inf(sigma12, "sigma12")
    L = 1/3
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2 

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # check_nan_inf(ssim_map, "ssim_map")

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss_v2(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        _ssim_val = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return _ssim_val

##########################################################################
# **Edge Loss (基於論文版本)**
class EdgeLoss_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:,:,::2,::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4   # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        x, y = self.laplacian_kernel(x), self.laplacian_kernel(y)
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss