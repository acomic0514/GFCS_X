import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.dataloader import RainDataset, get_transform
import pandas as pd

# Import modules
from models.archs.DPENet_v1 import DPENet
from models.archs.DPENet_v2 import DPENet_CFIM
from models.archs.DPENet_v3 import DPENet_CFIM_v2
from models.archs.DPE_Unet import DPE_Unet
from models.archs.losses import SSIMLoss_v2, EdgeLoss_v2, L1Loss
from models.CosineAnnealingRestartCyclicLR import CosineAnnealingRestartCyclicLR

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter
import datetime
from torchvision.utils import make_grid


#共用函數區
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Ensure specific types
    config["epochs"] = int(config["epochs"])
    config["batch_size"] = int(config["batch_size"])
    config["lr"] = float(config["lr"])
    config["eta_min"] = float(config["eta_min"])
    config["periods"] = [int(period) for period in config["periods"]]
    config["restart_weights"] = [float(weight) for weight in config["restart_weights"]]
    config["num_workers"] = int(config["num_workers"])
    config["log_interval"] = int(config["log_interval"])
    config["use_amp"] = bool(config["use_amp"])
    config["grad_clip"] = float(config["grad_clip"])  # 新增梯度裁剪閾值，預設為 1.0
    
    return config

"""
def _visualize_results(input_img, mid_output, final_output, target_img, title_prefix=""):
    def to_numpy(img):
        return img[0].clamp(0, 1).cpu().detach().permute(1, 2, 0).numpy()

    images = {
        f"{title_prefix}_input": to_numpy(input_img),
        f"{title_prefix}_mid_output": to_numpy(mid_output),
        f"{title_prefix}_final_output": to_numpy(final_output),
        f"{title_prefix}_target": to_numpy(target_img),
    }

    fig, axes = plt.subplots(1, 4, figsize=(2.56*4, 2.56), dpi=100)
    for idx, (title, img) in enumerate(images.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()
"""
   
# def log_sample_to_tensorboard(writer, input_img, mid_output, final_output, target_img, epoch, prefix="Train"):
def log_sample_to_tensorboard(writer, input_img, mid_output, target_img, epoch, prefix="Train"):
    def preprocess(tensor):
        return tensor.clamp(0, 1).detach().cpu()

    # 取 batch 中第 0 張圖，並展開成 4 張 [C, H, W]
    imgs = [
        preprocess(input_img[0]),
        preprocess(mid_output[0]),
        # preprocess(final_output[0]),
        preprocess(target_img[0]),
    ]

    # 組成 grid [4*C, H, W] 並加一個 batch 維度
    # grid = make_grid(imgs, nrow=4)  # [C, H, W]
    grid = make_grid(imgs, nrow=3)  # [C, H, W]
    writer.add_image(f"{prefix}/Sample_Epoch_{epoch}", grid, global_step=epoch)
    
def sliding_window_inference(img_tensor: torch.Tensor, model, patch_size=(128, 128), stride=(64, 64)) -> torch.Tensor:
    """
    img_tensor: (1, C, H, W)
    model: 可呼叫模型，輸入為 (B, C, H, W)
    return: (1, C, H, W) 與原圖等尺寸的推理結果
    """
    assert img_tensor.ndim == 4 and img_tensor.size(0) == 1, "請輸入 (1, C, H, W) 的 Tensor"
    B, C, H, W = img_tensor.shape
    ph, pw = patch_size
    sh, sw = stride

    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw

    pad_img = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = pad_img.shape

    output_mid = torch.zeros_like(pad_img)
    count_map_mid = torch.zeros_like(pad_img)
    output_final = torch.zeros_like(pad_img)
    count_map_final = torch.zeros_like(pad_img)

    # 輸出有兩個，分別是中間輸出與最終輸出
    # with torch.no_grad():
    #     for i in range(0, H_pad - ph + 1, sh):
    #         for j in range(0, W_pad - pw + 1, sw):
    #             patch = pad_img[:, :, i:i+ph, j:j+pw]
    #             pred_mid, pred_final = model(patch)  # 假設 output shape 與 input 相同
    #             pred_mid, pred_final = model(patch)  # 假設 output shape 與 input 相同
    #             output_mid[:, :, i:i+ph, j:j+pw] += pred_mid
    #             output_final[:, :, i:i+ph, j:j+pw] += pred_final
    #             count_map_mid[:, :, i:i+ph, j:j+pw] += 1
    #             count_map_final[:, :, i:i+ph, j:j+pw] += 1

    # output_mid = output_mid / count_map_mid
    # output_final = output_final / count_map_final
    # return output_mid[:, :, :H, :W], output_final[:, :, :H, :W] # 去除 padding

    # 輸出僅有最終輸出
    with torch.no_grad():
        for i in range(0, H_pad - ph + 1, sh):
            for j in range(0, W_pad - pw + 1, sw):
                patch = pad_img[:, :, i:i+ph, j:j+pw]
                pred_final = model(patch)  # 假設 output shape 與 input 相同
                output_final[:, :, i:i+ph, j:j+pw] += pred_final
                count_map_final[:, :, i:i+ph, j:j+pw] += 1

    output_final = output_final / count_map_final
    return output_final[:, :, :H, :W] # 去除 padding
    

# 訓練/驗證模組區
def train_one_epoch(model, dataloader, optimizer, loss_fns, device, config, epoch, writer=None):
    model.train()
    running_loss = 0.0
    running_mid_ssim = 0.0
    running_final_ssim = 0.0
    ssim_loss, edge_loss = loss_fns

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config['epochs']}]")
    for batch_idx, batch in enumerate(pbar):
        input_img = batch['input']
        target_img = batch['target']
        input_img, target_img = input_img.to(device), target_img.to(device)
        optimizer.zero_grad()

        # mid_output, final_output = model(input_img)
        final_output = model(input_img)
        
        # 帶平移縮放的輸出
        # mid_output_shift = (mid_output + 1) / 3
        # final_output_shift = (final_output + 1) / 3
        # target_img_shift = (target_img + 1) / 3
        # mid_ssim = ssim_loss(mid_output_shift, target_img_shift)
        # final_ssim = ssim_loss(final_output_shift, target_img_shift)
        # ssim_val = (1 - mid_ssim) + (1 - final_ssim)
        # edge_val = edge_loss(mid_output_shift, target_img_shift) + edge_loss(final_output_shift, target_img_shift)
        
        # 含中間輸出
        # mid_ssim = ssim_loss(mid_output, target_img)
        # final_ssim = ssim_loss(final_output, target_img)
        # ssim_val = (1 - mid_ssim) + (1 - final_ssim)
        # edge_val = edge_loss(mid_output, target_img) + edge_loss(final_output, target_img)
        
        # 僅最後輸出
        final_ssim = ssim_loss(final_output, target_img)
        ssim_val = (1 - final_ssim)
        edge_val = edge_loss(final_output, target_img)
        
        
        """
        # mid_output = model(input_img)
        mid_ssim = ssim_loss(mid_output, target_img)
        final_ssim = ssim_loss(final_output, target_img)
        ssim_val = (1 - mid_ssim) + (1 - final_ssim)
        # ssim_val = (1 - mid_ssim) 
        edge_val = edge_loss(mid_output, target_img) + edge_loss(final_output, target_img)
        # edge_val = edge_loss(mid_output, target_img) 
        """
        
        if torch.isnan(ssim_val) or torch.isinf(ssim_val):
            print("⚠ SSIM Loss 出錯", ssim_val, input_img.sum(), target_img.sum())
            print("input img 最大值:", input_img.max().item(), "最小值:", input_img.min().item())
            print("target img 最大值:", target_img.max().item(), "最小值:", target_img.min().item())
            # print("mid output img 最大值:", mid_output.max().item(), "最小值:", mid_output.min().item())

        if torch.isnan(edge_val) or torch.isinf(edge_val):
            print("⚠ Edge Loss 出錯", edge_val, input_img.sum(), target_img.sum())
            print("input img 最大值:", input_img.max().item(), "最小值:", input_img.min().item())
            print("target img 最大值:", target_img.max().item(), "最小值:", target_img.min().item())
            # print("mid output img 最大值:", mid_output.max().item(), "最小值:", mid_output.min().item())

        loss = ssim_val + 0.05 * edge_val + 0.2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        running_loss += loss.item()
        # running_mid_ssim += mid_ssim.item()
        running_final_ssim += final_ssim.item()
        if batch_idx % config["log_interval"] == 0:
            pbar.set_postfix(loss=loss.item())
            
        if writer is not None and (epoch + 1) % 10 == 0 and batch_idx == 0:
            # log_sample_to_tensorboard(writer, input_img, mid_output, final_output, target_img, epoch,
            #                           prefix=f"Epoch_{epoch+1}_Batch_{batch_idx+1}")
            log_sample_to_tensorboard(writer, input_img, final_output, target_img, epoch,
                                      prefix=f"Epoch_{epoch+1}_Batch_{batch_idx+1}")
                

    avg_loss = running_loss / len(dataloader)
    avg_mid_ssim = running_mid_ssim / len(dataloader)
    avg_final_ssim = running_final_ssim / len(dataloader)
    
    print(f"\t🔹 Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.6f}, mid SSIM: {avg_mid_ssim:.6f}, final SSIM: {avg_final_ssim:.6f}")
    return avg_loss, avg_mid_ssim, avg_final_ssim
    # return avg_loss, avg_mid_ssim

def validate_one_epoch(model, dataloader, loss_fns, device, config, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    running_mid_ssim = 0.0
    running_final_ssim = 0.0
    ssim_loss, edge_loss = loss_fns

    with torch.no_grad():
        for val_idx, batch in enumerate(dataloader):
            input_img = batch['input']
            target_img = batch['target']
            input_img, target_img = input_img.to(device), target_img.to(device)
            # mid_output, final_output = model(input_img)
            # mid_output = model(input_img)
            # mid_output, final_output = sliding_window_inference(input_img, model, patch_size=(128, 128), stride=(64, 64))
            final_output = sliding_window_inference(input_img, model, patch_size=(128, 128), stride=(64, 64))
            
            # 平移縮放的輸出
            # mid_output_shift = (mid_output + 1) / 3
            # target_img_mid_shift = (target_img + 1) / 3
            # final_output_shift = (final_output + 1) / 3
            # target_img_shift = (target_img + 1) / 3
            # mid_ssim_val = ssim_loss(mid_output_shift, target_img_mid_shift)
            # final_ssim_val = ssim_loss(final_output_shift, target_img_shift)
            # ssim_val = (1 - mid_ssim_val) + (1 - final_ssim_val)
            # edge_val = edge_loss(mid_output_shift, target_img_mid_shift) + edge_loss(final_output_shift, target_img_shift)
            
            # 含中間輸出
            # mid_ssim_val = ssim_loss(mid_output, target_img)
            # final_ssim_val = ssim_loss(final_output, target_img)
            # ssim_val = (1 - mid_ssim_val) + (1 - final_ssim_val)
            # edge_val = edge_loss(mid_output, target_img) + edge_loss(final_output, target_img)
                        
            # 僅最後輸出
            final_ssim_val = ssim_loss(final_output, target_img)
            ssim_val = (1 - final_ssim_val)
            edge_val =  edge_loss(final_output, target_img)
            
            loss = ssim_val + 0.05 * edge_val + 0.2

            """
            mid_ssim_val = ssim_loss(mid_output, target_img)
            # final_ssim_val = ssim_loss(final_output, target_img)
            # ssim_val = (1 - ssim_loss(mid_output, target_img)) + (1 - ssim_loss(final_output, target_img))
            ssim_val = (1 - ssim_loss(mid_output, target_img))
            # edge_val = edge_loss(mid_output, target_img) + edge_loss(final_output, target_img)
            edge_val = edge_loss(mid_output, target_img) 
            loss = ssim_val + 0.05 * edge_val + 0.2
            """
            
            running_loss += loss.item()
            # running_mid_ssim += mid_ssim_val.item()
            running_final_ssim += final_ssim_val.item()
            
            if writer is not None and (epoch + 1) % 10 == 0 and val_idx == 0:
                # log_sample_to_tensorboard(writer, input_img, mid_output, final_output, target_img, epoch,
                #                           prefix=f"Epoch_{epoch+1}_Batch_{val_idx+1}")
                log_sample_to_tensorboard(writer, input_img, final_output, target_img, epoch,
                                          prefix=f"Epoch_{epoch+1}_Batch_{val_idx+1}")

    avg_loss = running_loss / len(dataloader)
    avg_mid_ssim_val = running_mid_ssim / len(dataloader)
    avg_final_ssim_val = running_final_ssim / len(dataloader)
    print(f"\t🔸 Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.6f}, 驗證集 mid SSIM: {avg_mid_ssim_val:.6f}, final SSIM: {avg_final_ssim_val:.6f}")
    return avg_loss, avg_mid_ssim_val, avg_final_ssim_val
    # return avg_loss, avg_mid_ssim_val

#訓練腳本參數設定
# Load configuration from YAML file
config = load_config('config.yml')
    
# 創建存放模型的資料夾
os.makedirs(config["checkpoint_dir"], exist_ok=True)

# 創建模型
model = DPE_Unet()
model.to(config["device"])

# 創建損失函數
ssim_loss = SSIMLoss_v2().to(config["device"])
edge_loss = EdgeLoss_v2().to(config["device"])  
loss_fns = (ssim_loss, edge_loss)

# 創建優化器 & 學習率調整器
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[130, 150, 180], gamma=0.2)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 70], gamma=0.2) # 斷點接續用 +110 in fact

train_dataset = RainDataset(mode='train', dataset_name=config["dataset_name"], 
                            transform=get_transform(train=True), inference=False)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                        shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_dataset = RainDataset(mode='test', dataset_name=config["testset_name"], 
                          transform=get_transform(train=False), inference=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": []
    }
# TensorBoard 設定
log_subdir = f"tensorboard_logs_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_dir = os.path.join(config["checkpoint_dir"], log_subdir)
writer = SummaryWriter(log_dir=log_dir)
writer.add_text("Config", str(config))

# # checkpoint 檔案路徑
# checkpoint_path = os.path.join(config["checkpoint_dir"], "DPENet_epoch40.pth")  # 例如第 40 epoch 儲存的模型
# # 讀取 checkpoint 檔案
# checkpoint = torch.load(checkpoint_path)
# # 還原模型、優化器、排程器狀態
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# # 還原 epoch
# start_epoch = checkpoint['epoch'] + 1  # +1 是為了從下一 epoch 開始繼續
# for epoch in range(start_epoch, config["epochs"]): #斷點接續使用
#訓練迴圈
for epoch in range(config["epochs"]):
    # train_loss, train_mid_ssim = train_one_epoch(
    #     model, train_loader, optimizer, loss_fns, config["device"], config, epoch , writer=writer
    #     )
    # val_loss, val_mid_ssim = validate_one_epoch(
    #     model, val_loader, loss_fns, config["device"], config, epoch, writer=writer
    #     )
    train_loss, train_mid_ssim, train_final_ssim = train_one_epoch(
        model, train_loader, optimizer, loss_fns, config["device"], config, epoch , writer=writer
        )
    val_loss, val_mid_ssim, val_final_ssim = validate_one_epoch(
        model, val_loader, loss_fns, config["device"], config, epoch, writer=writer
        )

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)    
    writer.add_scalar('SSIM/Train_Mid', train_mid_ssim, epoch)
    writer.add_scalar('SSIM/Train_Final', train_final_ssim, epoch)    
    writer.add_scalar('SSIM/Val_Mid', val_mid_ssim, epoch)
    writer.add_scalar('SSIM/Val_Final', val_final_ssim, epoch)
    writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
    writer.add_scalars('SSIM/Mid', {'Train': train_mid_ssim, 'Val': val_mid_ssim}, epoch)
    writer.add_scalars('SSIM/Final', {'Train': train_final_ssim, 'Val': val_final_ssim}, epoch)
    
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        # 'loss': train_loss,  # 若要追蹤}
        }
        # Save the model checkpoint
        save_path = os.path.join(config["checkpoint_dir"], f"DPENet_epoch{epoch+1}.pth")
        torch.save(checkpoint, save_path)
        print(f"模型已保存：{save_path}")
        
# 實驗結果儲存
# Save training history to CSV
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(config["checkpoint_dir"], "training_history.csv"), index=False)
print(f'✅ 訓練歷程已儲存至：{config["checkpoint_dir"]}')

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss", marker='o')
plt.plot(history["epoch"], history["val_loss"], label="Val Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Curve")
plt.grid(True)
plt.legend()
combined_loss_plot_path = os.path.join(config["checkpoint_dir"], "train_val_loss_curve.png")
plt.savefig(combined_loss_plot_path)
plt.close()
print(f"📉 Loss 曲線（Train vs Val）已儲存：{combined_loss_plot_path}")

"""
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from metrics.dataloader import RainDataset, get_transform

# Import modules
from models.archs.DPENet_v1 import DPENet
from models.archs.losses import SSIMLoss_v2, EdgeLoss_v2, L1Loss
from models.CosineAnnealingRestartCyclicLR import CosineAnnealingRestartCyclicLR

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Ensure specific types
    config["epochs"] = int(config["epochs"])
    config["batch_size"] = int(config["batch_size"])
    config["lr"] = float(config["lr"])
    config["eta_min"] = float(config["eta_min"])
    config["periods"] = [int(period) for period in config["periods"]]
    config["restart_weights"] = [float(weight) for weight in config["restart_weights"]]
    config["num_workers"] = int(config["num_workers"])
    config["log_interval"] = int(config["log_interval"])
    config["use_amp"] = bool(config["use_amp"])
    config["grad_clip"] = float(config["grad_clip"])  # 新增梯度裁剪閾值，預設為 1.0
    
    return config

def denormalize(tensor, mean, std):

    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)  # 調整 shape 以匹配輸入
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean  # 反歸一化

if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config('config.yml')
        
    # 創建存放模型的資料夾
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # 創建模型
    model = DPENet()
    model.to(config["device"])

    # 創建損失函數
    ssim_loss = SSIMLoss_v2().to(config["device"])
    edge_loss = EdgeLoss_v2().to(config["device"])
    l1_loss = L1Loss(loss_weight=1.0, reduction='mean').to(config["device"])

    # 創建優化器 & 學習率調整器
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], 
                            betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingRestartCyclicLR(optimizer, periods=config["periods"], 
                                            restart_weights=config["restart_weights"], 
                                            eta_mins=[config["lr"], config["eta_min"]])

    # 設定 DataLoader
    train_dataset = RainDataset(mode='train', dataset_name=config["dataset_name"], 
                                transform=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], 
                                            shuffle=True, num_workers=4)

    # 設定 AMP（混合精度）
    scaler = GradScaler('cuda', enabled=config["use_amp"])
    # 訓練迴圈
    print(f"開始訓練 DPENet_CFIM，使用設備：{config['device']}")
    
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['epochs']}]")

        for batch_idx, (input_img, target_img) in enumerate(pbar):
            input_img, target_img = input_img.to(config["device"]), target_img.to(config["device"])

            optimizer.zero_grad()

            # 前向傳播（AMP 混合精度）
            with autocast(device_type='cuda', enabled=config["use_amp"]):
                output = model(input_img)
                ssim_val = ssim_loss(output, target_img)
                edge_val = edge_loss(output, target_img)
                if torch.isnan(ssim_val) or torch.isinf(ssim_val):
                    print("⚠ SSIM Loss 出錯", ssim_val)

                if torch.isnan(edge_val) or torch.isinf(edge_val):
                    print("⚠ Edge Loss 出錯", edge_val)

                loss = ssim_val + 0.05 * edge_val

            # 反向傳播
            scaler.scale(loss).backward()
            
            # 梯度裁剪，防止梯度爆炸或震盪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 更新學習率

            # 記錄 Loss
            running_loss += loss.item()
            if batch_idx % config["log_interval"] == 0:
                pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"🔹 Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.6f}, LR: {scheduler.get_lr()[0]:.8f}")

        # 保存模型（每 10 個 Epoch 保存一次）
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config["checkpoint_dir"], f"DPENet_CFIM_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存：{save_path}")

"""