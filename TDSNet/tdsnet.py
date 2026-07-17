import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math


class SpectralPhaseAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # 
        self.phase_filter = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x_norm = self.norm(x)
        x_spatial = x_norm.transpose(1, 2).reshape(b, c, h, w)

        fft_x = torch.fft.rfft2(x_spatial.float(), norm='ortho')
        
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        phase_features = torch.cat([sin_phase, cos_phase], dim=1) # [B, 2C, H, W/2+1]
        
        phase_mask = self.phase_filter(phase_features)
        
        # 将掩码作用于振幅，并结合原始相位重建复数
        refined_fft = amplitude * phase_mask * torch.exp(1j * phase)
        
        # 逆傅里叶变换回空域
        x_refined = torch.fft.irfft2(refined_fft, s=(h, w), norm='ortho')
        x_refined = x_refined.reshape(b, c, -1).transpose(1, 2)
        
        return x + self.spatial_proj(x_refined)

class DynamicTopologyGraph(nn.Module):
    def __init__(self, dim, k=9):
        super().__init__()
        self.k = k 
        self.query_proj = nn.Linear(dim, dim // 2)
        self.key_proj = nn.Linear(dim, dim // 2)
        self.value_proj = nn.Linear(dim, dim)
        self.outlier_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        
        self.gamma = nn.Parameter(1e-4 * torch.ones((dim)), requires_grad=True)
        
    def forward(self, x):
        b, n, c = x.shape
        q = self.query_proj(x)
        k = self.key_proj(x)  
        v = self.value_proj(x)
        
        attn = torch.bmm(q, k.transpose(1, 2)) / (c ** 0.5) 
        
        mask = torch.zeros_like(attn)
        topk_val, topk_idx = torch.topk(attn, self.k, dim=-1)
        mask.scatter_(-1, topk_idx, 1.0)
        attn = torch.where(mask > 0, attn, torch.tensor(-1e9).to(attn.device))
        attn_weights = F.softmax(attn, dim=-1)
        
        graph_msg = torch.bmm(attn_weights, v) 
        outlier_residual = x - graph_msg
        
        outlier_feature = outlier_residual * torch.sigmoid(self.outlier_gate(outlier_residual))
        
        return x + outlier_feature * self.gamma


class DeformableUpsample(nn.Module):
    def __init__(self, in_dim, out_dim, max_shift=3.0):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        self.max_shift = max_shift 

        laplacian = torch.tensor([[-1., -1., -1.],
                                  [-1.,  8., -1.],
                                  [-1., -1., -1.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian)
        
        self.artifact_analyzer = nn.Sequential(
            nn.Conv2d(out_dim + 1, out_dim // 2, 3, padding=1),
            nn.GELU(),

            nn.Conv2d(out_dim // 2, 2, 3, padding=1) 
        )
        
       
        self.offset_pred = nn.Sequential(
           
            nn.Conv2d(out_dim, out_dim, 5, padding=2, groups=out_dim), 
            nn.Conv2d(out_dim, out_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(out_dim // 2, 2, 3, padding=1)
        )
        
        nn.init.constant_(self.offset_pred[-1].weight, 0)
        nn.init.constant_(self.offset_pred[-1].bias, 0)
        

        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_proj = self.proj(x)
        c_out = x_proj.shape[1]
        

        hr_base = F.interpolate(x_proj, scale_factor=2, mode='bilinear', align_corners=False)
        B, C, H_new, W_new = hr_base.shape
        

        gray_base = hr_base.mean(dim=1, keepdim=True) 
        high_freq = F.conv2d(gray_base, self.laplacian_kernel.to(gray_base.dtype), padding=1)
        

        analyzer_in = torch.cat([hr_base, high_freq], dim=1)
        analyzer_out = self.artifact_analyzer(analyzer_in)
        

        artifact_mask = torch.sigmoid(analyzer_out[:, 0:1, ...]) # [B, 1, 2H, 2W]
        modulation_mask = torch.sigmoid(analyzer_out[:, 1:2, ...]) # [B, 1, 2H, 2W]
        
       
        raw_offsets = self.offset_pred(hr_base) # [B, 2, 2H, 2W]
        
        offset_norm_x = torch.tanh(raw_offsets[:, 0:1, ...]) * self.max_shift * (2.0 / W_new)
        offset_norm_y = torch.tanh(raw_offsets[:, 1:2, ...]) * self.max_shift * (2.0 / H_new)
        bounded_offsets = torch.cat([offset_norm_x, offset_norm_y], dim=1)
        
        gated_offsets = bounded_offsets * artifact_mask
        
        device = x.device
        theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float, device=device).expand(b, -1, -1)
        base_grid = F.affine_grid(theta, size=(b, c_out, H_new, W_new), align_corners=False) 
        
        deform_grid = base_grid + gated_offsets.permute(0, 2, 3, 1)
        
        deform_up = F.grid_sample(x_proj, deform_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        deform_up = deform_up * modulation_mask

        out = deform_up + self.refine(deform_up)
        
        return out

class GS3Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spectral_block = SpectralPhaseAttention(dim)
        self.graph_block = DynamicTopologyGraph(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x, h, w):
        x = self.spectral_block(x, h, w)
        x = self.graph_block(x)
        x = x + self.ffn(x)
        return x


class GS3Decoder(nn.Module):
    
    def __init__(self, embed_dim=384, num_layers=4, out_channels=[256, 128, 64, 32], nclass=1):
        super().__init__()

        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_layers, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels[0])
        )
        
        self.layers = nn.ModuleList([
            GS3Layer(out_channels[0]) for _ in range(3)
        ])
        
        self.upsamplers = nn.ModuleList([
            DeformableUpsample(out_channels[0], out_channels[1]), # 32 -> 64
            DeformableUpsample(out_channels[1], out_channels[2]), # 64 -> 128
            DeformableUpsample(out_channels[2], out_channels[3]), # 128 -> 256
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[3], 16, 4, 2, 1), 
                nn.GELU()
            )
        ])
        
        self.head = nn.Conv2d(16, nclass, kernel_size=1)

    def forward(self, features, patch_h, patch_w):
        processed_features = []
        
        for f in features:
            if f.ndim == 3:
                if f.shape[1] == patch_h * patch_w + 1:
                    f = f[:, 1:, :] 
                elif f.shape[1] != patch_h * patch_w:
                    raise ValueError(f"Token分辨率不匹配")
            elif f.ndim == 4:
                f = f.flatten(2).transpose(1, 2)
            processed_features.append(f)
            
        # [B, N, C*4]
        x_concat = torch.cat(processed_features, dim=-1)
        
        # [B, N, out_channels[0]]
        x = self.feature_fusion(x_concat)
        
        for layer in self.layers:
            x = layer(x, patch_h, patch_w)
            
        # [B, C, H, W]
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, patch_h, patch_w)
        
        for up in self.upsamplers:
            x = up(x)
            
        return self.head(x)

class DPT(nn.Module):
    def __init__(self, encoder_size='base', backbone='None', nclass=1):
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
        self.decoder = GS3Decoder(
            embed_dim=self.backbone.embed_dim, 
            num_layers=len(self.intermediate_layer_idx), 
            nclass=nclass
        )

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=self.intermediate_layer_idx[self.encoder_size])
        mask = self.decoder(features, x.shape[-2]//16, x.shape[-1]//16)
        return mask