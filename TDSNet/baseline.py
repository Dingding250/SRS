import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math


# class GS3Layer(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.ffn = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Linear(dim * 2, dim)
#         )

#     def forward(self, x, h, w):
#         x = x + self.ffn(x)
#         return x


class Decoder(nn.Module):
    # 修改：增加了 num_layers 参数，用于特征拼接降维
    def __init__(self, embed_dim=384, num_layers=4, out_channels=[256, 128, 64, 32], nclass=1):
        super().__init__()
        
        # --- 核心修改：多层特征拼接融合 ---
        # 接收 backbone 的多层特征 (如层 2,5,8,11)，拼接后降维
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_layers, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels[0])
        )
        
        # self.layers = nn.ModuleList([
        #     GS3Layer(out_channels[0]) for _ in range(2)
        # ])
        
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[0], out_channels[1], 4, 2, 1), 
                nn.GELU(),
                nn.ConvTranspose2d(out_channels[1], out_channels[2], 4, 2, 1), 
                nn.GELU(),
                nn.ConvTranspose2d(out_channels[2], out_channels[3], 4, 2, 1), 
                nn.GELU(),
                nn.ConvTranspose2d(out_channels[3], 16, 4, 2, 1), 
                nn.GELU()
            )
        ])
        
        self.head = nn.Conv2d(16, nclass, kernel_size=1)

    def forward(self, features, patch_h, patch_w):
        processed_features = []
        
        # --- 核心修改：遍历所有传入的层特征 ---
        for f in features:
            processed_features.append(f)
            
        # [B, N, C*4]
        x_concat = torch.cat(processed_features, dim=-1)
        
        # 融合浅层(细节)与深层(语义) -> [B, N, out_channels[0]]
        x = self.feature_fusion(x_concat)
        
        # for layer in self.layers:
        #     x = layer(x, patch_h, patch_w)
            
        # [B, C, H, W]
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, patch_h, patch_w)
        
        for up in self.upsamplers:
            x = up(x)
            
        # 3. 输出伪影Mask
        return self.head(x)

class DPT(nn.Module):
    def __init__(self, encoder_size='large', backbone='None', nclass=1):
        super().__init__()
        self.backbone = backbone
        self.encoder_size = encoder_size
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        self.decoder = Decoder(
            embed_dim=self.backbone.embed_dim, 
            num_layers=len(self.intermediate_layer_idx), # 传入融合的层数
            nclass=nclass
        )

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder_size])
        mask = self.decoder(features, x.shape[-2]//16, x.shape[-1]//16)
        return mask