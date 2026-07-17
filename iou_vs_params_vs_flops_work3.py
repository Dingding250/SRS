import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# **********************************************
# ** 修改点 1：设置全局字体为 Arial **
# **********************************************
# 检查系统是否支持 Arial 字体
# 如果系统中没有 Arial 字体，Matplotlib 会回退到默认字体。
# 您也可以使用 font.sans-serif 来指定多个备选字体。
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif'] # 将 Arial 作为首选

# --- 1. 准备数据 ---
data = {
    'Methods': [
        'HiFi-Net++', 'PAL4Inpaint', 'PSCC-Net', 'TruFor', 'DeSRA', 
        'SeaFormer++', 'SegFormer-B4', 'K-Net', 'SegFormer-B5', 'PAL4VST', 
        'LRFormer', 'Mask2Former', 'LRFormer+', 'TDSNet-S', 'TDSNet-B', 'TDSNet-L'
    ],
    'Params(M)': [
        6.89, 48.96, 3.67, 68.70, 3.72, 
        6.37, 61.37, 72.15, 81.97, 120.00, 
        32.17, 88.90, 48.00, 23.11, 85.99, 295.22
    ],
    'Flops(G)': [
        248.78, 150, 102.65, 221.83, 7.89,  # PAL4Inpaint 的 Flops 为 '-', 使用 np.nan
        13.92, 58.57, 254.98, 74.55, 305.00, 
        39.75, 139.56, 70.00, 25.74, 77.16, 246.95
    ],
    'IoU (%)': [
        0.36, 2.66, 3.03, 3.52, 6.12, 
        10.54, 21.42, 22.70, 26.17, 29.28, 
        31.19, 33.28, 34.32, 38.02, 40.84, 41.36
    ]
}

df = pd.DataFrame(data)

# **自定义颜色列表**
custom_colors = {
    'HiFi-Net++': 'darkorange',
    'PAL4Inpaint': 'gold',
    'PSCC-Net': 'cornflowerblue',
    'TruFor': 'limegreen',
    'DeSRA': 'hotpink',
    'SeaFormer++': 'indigo',
    'SegFormer-B4': 'mediumslateblue',
    'K-Net': 'darkorange',
    'SegFormer-B5': 'lightseagreen',
    'PAL4VST': 'turquoise',
    'LRFormer': 'deepskyblue',
    'Mask2Former': 'dodgerblue',
    'LRFormer+': 'orchid',
    'TDSNet-S': 'red',
    'TDSNet-B': 'red',
    'TDSNet-L': 'red',
}
face_colors = [custom_colors[method] for method in df['Methods']]

# --- 2. 设置气泡大小 (Model Size) ---
size_factor = 500
sizes = np.power(df['Params(M)'], 1.5) * size_factor / df['Params(M)'].max()

# --- 3. 绘图 ---
plt.figure(figsize=(12*0.7, 8*0.7))

# 绘制气泡图
scatter = plt.scatter(
    df['Flops(G)'],      # X轴: Flops(G)
    df['IoU (%)'],       # Y轴: IoU (%)
    s=sizes,             # 气泡大小: Params(M)
    alpha=0.5,           # 略微增加透明度
    linewidths=2.0,
    c=face_colors        # 使用自定义填充颜色列表
)

# 添加数据标签 (即Methods名称)
for i, row in df.iterrows():
    # 为了避免重叠，根据气泡大小和IoU值进行更精细的偏移调整
    x_offset = 3 if row['IoU (%)'] < 10 else 5 # IoU低的点稍微减少x偏移
    y_offset = 0.5 if row['Methods'] != 'HiFi-Net' else +1 # HiFi-Net在底部，向上偏移
    
    plt.annotate(
        row['Methods'],
        (row['Flops(G)'] + x_offset, row['IoU (%)'] + y_offset),
        fontsize=10,
        ha='left',
        fontname='Arial'
    )

# --- 4. 添加图表元素 ---

plt.title('IoU (%) vs. Flops (G) vs. Params (M)', fontsize=15, pad=15, fontname='Arial')
plt.xlabel('Flops (G)', fontsize=15, fontname='Arial')
plt.ylabel('IoU (%)', fontsize=15, fontname='Arial')

# 刻度标签的字体也会受到全局设置的影响
plt.tick_params(axis='both', which='major', labelsize=12) # 调整刻度字体大小

plt.grid(True, linestyle='-', alpha=1.0)

# 限制轴范围 (确保所有数据点可见)
plt.xlim(0, df['Flops(G)'].max() * 1.1)
plt.ylim(0, df['IoU (%)'].max() * 1.1)

# 添加 Model Size 图例
legend_params = [5, 50, 80, 150]
legend_sizes = np.power(np.array(legend_params), 1.5) * size_factor / df['Params(M)'].max()

legend_elements = [
    # 图例气泡
    plt.scatter([], [], s=size, color='lightgray', alpha=0.5, edgecolors='gray', linewidths=1.5, label=f'{param}M')
    for param, size in zip(legend_params, legend_sizes)
]

legend = plt.legend(
    legend_elements,
    [f'{param}M' for param in legend_params],
    title='Params(M)',
    loc='center right', # 位置：右侧中间
    bbox_to_anchor=(1, 0.5), # 微调位置到右边缘
    frameon=True,
    borderpad=3,
    labelspacing=2,
    ncol=3, # 更改为1列，使图例更紧凑
    fontsize=12,
    title_fontsize=14
)

# 显式设置图例标题和标签的字体
for text in legend.get_texts():
    text.set_fontname("Arial")
legend.get_title().set_fontname("Arial")

plt.tight_layout(rect=[0, 0, 0.95, 1]) 
plt.show()