import os
from PIL import Image

def convert_png_to_jpg(folder_a, folder_b):
    """
    将文件夹A中的所有.png图片转换为.jpg格式并保存到文件夹B
    
    参数:
        folder_a: 源文件夹路径（包含.png图片）
        folder_b: 目标文件夹路径（保存.jpg图片）
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)
        print(f"已创建目标文件夹: {folder_b}")
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(folder_a):
        # 检查文件是否为.png格式
        if filename.lower().endswith('.png'):
            # 构建完整的文件路径
            png_path = os.path.join(folder_a, filename)
            
            # 构建目标.jpg文件路径（保持原文件名）
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(folder_b, jpg_filename)
            
            try:
                # 打开PNG图片并转换为JPG
                with Image.open(png_path) as img:
                    # 如果PNG有透明通道，转换为RGB模式（JPG不支持透明）
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                        background.paste(img, img.split()[-1])
                        img = background
                    elif img.mode == 'P':
                        img = img.convert('RGB')
                    
                    # 保存为JPG格式，质量设为95（0-100）
                    img.save(jpg_path, 'JPEG', quality=100)
                print(f"已转换: {filename} -> {jpg_filename}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")

if __name__ == "__main__":
    # 请根据实际情况修改以下文件夹路径
    folder_a = "D:\\segdino-main\\segdata\\unified\\train\\image"  # 源文件夹A路径
    folder_b = "D:\\segdino-main\\segdata\\nss\\train\\image"  # 目标文件夹B路径
    
    print(f"开始转换 {folder_a} 中的PNG图片到 {folder_b} 文件夹...")
    convert_png_to_jpg(folder_a, folder_b)
    print("转换完成！")