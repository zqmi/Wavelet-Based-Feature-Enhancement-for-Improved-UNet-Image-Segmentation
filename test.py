import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.loaders import RadioUNet_c
from lib.model import UNet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import imageio.v2 as imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试集
dataset_test = RadioUNet_c(dir_dataset='/home/disk01/qmzhang/RadioMapSeer/', phase='test')
dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4)

# 初始化模型
model = UNet(in_channels=1, out_channels=1).to(device)

# 加载模型权重，替换成你的模型路径
model_path = "checkpoints/unet_step90000.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

os.makedirs("vis_test", exist_ok=True)

# 指标累积器
nmse_total = 0.0
rmse_total = 0.0
ssim_total = 0.0
psnr_total = 0.0
count = 0

for outputs, inputs, name2 in tqdm(dataloader_test):
    inputs = inputs.to(device)        # [B, 1, H, W]
    targets = outputs.to(device)      # [B, 1, H, W]

    with torch.no_grad():
        preds = model(inputs)
        preds = torch.clamp(preds, 0, 1)

    preds_np = preds.cpu().numpy()      # [B, 1, H, W]
    targets_np = targets.cpu().numpy()  # [B, 1, H, W]
    inputs_np = inputs.cpu().numpy()    # [B, 1, H, W]

    B = preds_np.shape[0]
    for i in range(B):
        pred = preds_np[i, 0]
        target = targets_np[i, 0]
        input_img = inputs_np[i, 0]

        # 计算指标
        mse = np.mean((pred - target) ** 2)
        norm = np.sum(target ** 2)
        nmse = np.sum((pred - target) ** 2) / norm
        rmse = np.sqrt(mse)
        ssim_val = ssim(target, pred, data_range=1)
        psnr_val = psnr(target, pred, data_range=1)

        nmse_total += nmse
        rmse_total += rmse
        ssim_total += ssim_val
        psnr_total += psnr_val
        count += 1

        # 保存图片（uint8格式，0-255）
        base_name = os.path.splitext(name2[i])[0]
        imageio.imwrite(f"vis_test/{base_name}_input.png", (input_img * 255).astype(np.uint8))
        imageio.imwrite(f"vis_test/{base_name}_pred.png", (pred * 255).astype(np.uint8))
        imageio.imwrite(f"vis_test/{base_name}_gt.png", (target * 255).astype(np.uint8))

# 计算平均指标
print(f"Test NMSE: {nmse_total / count:.6f}")
print(f"Test RMSE: {rmse_total / count:.6f}")
print(f"Test SSIM: {ssim_total / count:.6f}")
print(f"Test PSNR: {psnr_total / count:.6f}")
