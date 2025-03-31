def check_nan_inf(self, x, comment=""):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("⚠️ 發現 NaN 或 Inf，檢查哪張圖片有問題...")
        
        # 找出包含 NaN 或 Inf 的圖片索引
        problematic_indices = []
        for i in range(x.shape[0]):
            if torch.isnan(x[i]).any() or torch.isinf(x[i]).any():
                problematic_indices.append(i)
        print(f"❌ comment: {comment}")
        print(f"❌ 有問題的圖片索引：{problematic_indices}")
        print(f"❌ 有問題得圖片數值為：{x[problematic_indices]}")
        
        return True