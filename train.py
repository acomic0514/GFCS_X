import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from metrics.dataloader import RainDataset, get_transform

# Import modules
from models.archs.GFCS_X_mod1 import GFCSNetwork
from models.archs.losses import L1Loss
from models.CosineAnnealingRestartCyclicLR import CosineAnnealingRestartCyclicLR

# 設定超參數
CONFIG = {
    "epochs": 100,  # 訓練輪數
    "batch_size": 8,  # 每批次處理的圖片數
    "lr": 3e-4,  # 初始學習率
    "eta_min": 1e-6,  # 最小學習率
    "periods": [92000, 208000],  # 餘弦衰減週期
    "restart_weights": [1, 1],  # 週期重啟時的學習率比例
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,  # DataLoader 進程數
    "checkpoint_dir": "checkpoints",  # 存放模型
    "log_interval": 10,  # 幾個 batch 記錄一次 Loss
    "use_amp": True,  # 是否使用混合精度
}

if __name__ == "__main__":
    # 創建存放模型的資料夾
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # 創建模型
    model = GFCSNetwork(inp_channels=3, out_channels=3, dim=48)
    model.to(CONFIG["device"])
    model.half()  # ✅ 確保模型參數使用 float16

    # 創建損失函數
    criterion = L1Loss().to(CONFIG["device"])

    # 創建優化器 & 學習率調整器
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], 
                            betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingRestartCyclicLR(optimizer, periods=CONFIG["periods"], 
                                            restart_weights=CONFIG["restart_weights"], 
                                            eta_mins=[CONFIG["lr"], CONFIG["eta_min"]])

    # 設定 DataLoader
    train_dataset = RainDataset(mode='train', dataset_name='Rain13K', 
                                transform=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, 
                                            shuffle=True, num_workers=4)

    # 設定 AMP（混合精度）
    scaler = GradScaler(enabled=CONFIG["use_amp"])
    
    # 訓練迴圈
    print(f"開始訓練 GFCS_X，使用設備：{CONFIG['device']}")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}]", ncols=100)

        for batch_idx, (input_img, target_img) in enumerate(pbar):
            input_img, target_img = input_img.to(CONFIG["device"]).half(), target_img.to(CONFIG["device"]).half()  # ✅ 確保輸入是 float16

            optimizer.zero_grad()

            # 前向傳播（AMP 混合精度）
            with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"]):  # ✅ 在 AMP 模式下前向傳播
                output = model(input_img)
                loss = criterion(output, target_img)

            # 反向傳播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 更新學習率

            # 記錄 Loss
            running_loss += loss.item()
            if batch_idx % CONFIG["log_interval"] == 0:
                pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"🔹 Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {avg_loss:.6f}, LR: {scheduler.get_lr()[0]:.8f}")

        # 保存模型（每 10 個 Epoch 保存一次）
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(CONFIG["checkpoint_dir"], f"GFCS_X_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存：{save_path}")

