import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt  # python小波库
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """两次卷积：Conv + BN + ReLU，常用UNet模块"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        prev_ch = in_channels
        for feature in features:
            self.encoder_layers.append(DoubleConv(prev_ch, feature))
            prev_ch = feature
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.upconvs.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder_layers.append(DoubleConv(feature*2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        enc_outs = []
        for enc in self.encoder_layers:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            enc_out = enc_outs[-(idx+1)]
            if x.shape != enc_out.shape:
                x = self._crop(enc_out, x)
            x = torch.cat([enc_out, x], dim=1)
            x = self.decoder_layers[idx](x)
        
        x = self.final_conv(x)
        return x
    
    def _crop(self, enc_feature, x):
        _, _, H, W = x.shape
        # 简单裁剪enc_feature中间部分以匹配x大小
        _, _, H_enc, W_enc = enc_feature.shape
        delta_H = H_enc - H
        delta_W = W_enc - W
        enc_feature = enc_feature[:,:, delta_H//2:H_enc - (delta_H - delta_H//2), delta_W//2:W_enc - (delta_W - delta_W//2)]
        return enc_feature

def haar_wavelet_features(x):
    """
    输入x形状 (B,1,H,W), 单通道图像
    返回多尺度小波特征：拼接原图 + LL,LH,HL,HH(放大到原始尺寸)
    输出形状 (B,5,H,W)
    """
    B, C, H, W = x.shape
    device = x.device
    x_np = x.cpu().numpy()  # 转numpy处理

    out_list = []
    for b in range(B):
        img = x_np[b,0]
        coeffs2 = pywt.dwt2(img, 'haar')  # 1级haar分解
        LL, (LH, HL, HH) = coeffs2
        
        # 转tensor并放回设备，尺寸缩小一半了(128×128)
        LL_t = torch.tensor(LL, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        LH_t = torch.tensor(LH, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        HL_t = torch.tensor(HL, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        HH_t = torch.tensor(HH, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        # 放大到原始大小256×256（双线性插值）
        LL_up = F.interpolate(LL_t, size=(H,W), mode='bilinear', align_corners=False)
        LH_up = F.interpolate(LH_t, size=(H,W), mode='bilinear', align_corners=False)
        HL_up = F.interpolate(HL_t, size=(H,W), mode='bilinear', align_corners=False)
        HH_up = F.interpolate(HH_t, size=(H,W), mode='bilinear', align_corners=False)

        # 拼接 原图 + 4个小波子带
        img_t = x[b:b+1]  # (1,1,H,W)
        cat_t = torch.cat([img_t, LL_up, LH_up, HL_up, HH_up], dim=1)  # (1,5,H,W)
        out_list.append(cat_t)
    
    out = torch.cat(out_list, dim=0)  # (B,5,H,W)
    return out

if __name__ == "__main__":
    import pywt
    model = UNet(in_channels=5, out_channels=1)
    x = torch.randn((1,1,256,256))
    x_wave = haar_wavelet_features(x)
    out = model(x_wave)
    print(f"输入拼接后尺寸: {x_wave.shape}")  # (1,5,256,256)
    print(f"输出尺寸: {out.shape}")  # (1,1,256,256)
