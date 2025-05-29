import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import models
import torch.optim as optim
import imageio.v2 as imageio 
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
import numpy as np
import os

from diffusers import AutoencoderKL

from rotary_embedding_torch import RotaryEmbedding

from lib.loaders import RadioUNet_c

from lib.model import UNet

from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

dataset = RadioUNet_c(dir_dataset='/home/disk01/qmzhang/RadioMapSeer/', phase='train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for batch in dataloader:
    outputs, inputs, name2 = batch

    # inputs: Tensor[B, C, H, W]
    # image_gain: Tensor[B, 1, H, W]
    # name2: list of strings (长度 B)

    print(outputs.shape)     #torch.Size([4, 1, 256, 256])
    print(inputs.shape)     #torch.Size([4, 1, 256, 256])
    print(name2)        #('282_24.png', '361_23.png', '339_72.png', '291_19.png')
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # 混合精度训练

num_epochs = 20

step = 0  # 全局 step 计数器

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for outputs, inputs, name2 in tqdm(dataloader):
        inputs = inputs.to(device)
        targets = outputs.to(device)

        optimizer.zero_grad()

        with autocast():
            preds = model(inputs)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        step += 1

        # 每1000步保存模型并执行一次推理
        if step % 10000 == 0:
            model_path = f"checkpoints/unet_step{step}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved checkpoint at step {step}: {model_path}")

            # 推理可视化
            model.eval()
            with torch.no_grad():
                sample_input = inputs[0:1]  # shape: [1, 1, 256, 256]
                sample_pred = model(sample_input)
                sample_target = targets[0:1]

                pred_np = sample_pred.cpu().squeeze().clamp(0, 1).numpy()
                target_np = sample_target.cpu().squeeze().numpy()
                input_np = sample_input.cpu().squeeze().numpy()

                os.makedirs("vis", exist_ok=True)

                # 保存图像用于观察推理效果
                imageio.imwrite(f"vis/infer_step{step}_input.png", (input_np * 255).astype(np.uint8))
                imageio.imwrite(f"vis/infer_step{step}_pred.png", (pred_np * 255).astype(np.uint8))
                imageio.imwrite(f"vis/infer_step{step}_gt.png", (target_np * 255).astype(np.uint8))
                print(f"🖼️  推理图像已保存至 vis/infer_step{step}_*.png")
            model.train()

    avg_loss = running_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")