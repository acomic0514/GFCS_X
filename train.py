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
    """
    反歸一化 Tensor 將 Normalize(mean, std) 轉回原始範圍
    """
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

