# 加载必要的库
import torch
from thop import profile, clever_format
from dpt import DPT

# 加载backbone并冻结
# backbone = torch.hub.load("./dinov3-main", 'dinov3_vits16', source='local', weights="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

# backbone = torch.hub.load("./dinov3-main", 'dinov3_vitb16', source='local', weights="./web_pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

backbone = torch.hub.load("./dinov3-main", 'dinov3_vitl16', source='local', weights="./web_pth/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

# ========== 关键步骤：冻结backbone的所有参数 ==========
for param in backbone.parameters():
    param.requires_grad = False

# 初始化DPT模型（包含冻结的backbone + 可训练的解码器）
model = DPT(backbone=backbone)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # 评估模式，避免BatchNorm/Dropout影响FLOPs计算

# print(model)

# ========== 1. 增强版参数量统计（区分backbone/解码器） ==========
def count_parameters_detailed(model, backbone):
    """
    详细统计参数量：
    - 整体模型（总/可训练/不可训练）
    - Backbone（总/不可训练，因为已冻结）
    - 解码器（总/可训练）
    """
    # 辅助函数：格式化参数数量
    def format_params(num):
        for unit in ['', 'K', 'M', 'G']:
            if abs(num) < 1024.0:
                return f"{num:.4f} {unit}Params"
            num /= 1024.0
        return f"{num:.4f} TParams"
    
    # 1. 统计整体模型
    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_all = total_all - trainable_all
    
    # 2. 统计backbone
    total_backbone = sum(p.numel() for p in backbone.parameters())
    # backbone已冻结，可训练参数为0
    trainable_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    non_trainable_backbone = total_backbone - trainable_backbone
    
    # 3. 统计解码器（整体模型 - backbone）
    total_decoder = total_all - total_backbone
    trainable_decoder = trainable_all - trainable_backbone  # 解码器全部可训练
    non_trainable_decoder = non_trainable_all - non_trainable_backbone  # 理论上为0
    
    return {
        # 整体模型
        'total_all': format_params(total_all),
        'trainable_all': format_params(trainable_all),
        'non_trainable_all': format_params(non_trainable_all),
        'total_all_raw': total_all,
        'trainable_all_raw': trainable_all,
        # Backbone
        'total_backbone': format_params(total_backbone),
        'trainable_backbone': format_params(trainable_backbone),
        'non_trainable_backbone': format_params(non_trainable_backbone),
        # 解码器
        'total_decoder': format_params(total_decoder),
        'trainable_decoder': format_params(trainable_decoder),
        'non_trainable_decoder': format_params(non_trainable_decoder)
    }

# 计算并打印详细参数量
param_stats = count_parameters_detailed(model, backbone)
print("="*60)
print("【冻结Backbone后】模型参数量详细统计:")
print("-"*30)
print("整体模型:")
print(f"  总参数量: {param_stats['total_all']}")
print(f"  可训练参数: {param_stats['trainable_all']} (仅解码器)")
print(f"  不可训练参数: {param_stats['non_trainable_all']} (仅backbone)")
print("-"*30)
print("Backbone（骨干网络）:")
print(f"  总参数量: {param_stats['total_backbone']}")
print(f"  可训练参数: {param_stats['trainable_backbone']} (已冻结)")
print(f"  不可训练参数: {param_stats['non_trainable_backbone']}")
print("-"*30)
print("解码器:")
print(f"  总参数量: {param_stats['total_decoder']}")
print(f"  可训练参数: {param_stats['trainable_decoder']} (核心训练部分)")
print(f"  不可训练参数: {param_stats['non_trainable_decoder']}")
print("="*60)

# ========== 2. 统计FLOPs（浮点运算量） ==========
# 注意：冻结参数仅影响反向传播时是否更新参数，不改变前向传播的FLOPs
input_tensor = torch.randn(1, 3, 512, 512).to(device)
flops, params = profile(model, inputs=(input_tensor,), verbose=False)

# 格式化输出
flops_formatted, params_formatted = clever_format([flops, params], "%.4f")
macs = flops / 2
macs_formatted = clever_format([macs], "%.4f")[0]

print("【冻结Backbone后】模型浮点运算量统计 (输入: 1,3,512,512):")
print(f"总FLOPs（前向计算量）: {flops_formatted}")
print(f"thop验证参数量: {params_formatted}")
print(f"乘加运算量(MACs): {macs_formatted}")
print("="*60)
