import os
from PIL import Image

def convert_jpg_to_png(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 分离文件名和扩展名（忽略大小写）
        name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()
        
        # 只处理.jpg和.jpeg格式的文件
        if ext_lower in ('.jpg', '.jpeg'):
            # 构建完整路径
            jpg_path = os.path.join(folder_path, filename)
            png_path = os.path.join(folder_path, f"{name}.png")
            
            try:
                # 打开图片并转换格式
                with Image.open(jpg_path) as img:
                    # 处理可能的alpha通道问题
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                        background.paste(img, img.split()[-1])
                        img = background
                    elif img.mode == 'P':
                        img = img.convert('RGB')
                    
                    # 保存为png格式
                    img.save(png_path, 'PNG')
                
                # 删除原jpg文件
                os.remove(jpg_path)
                # print(f"已转换: {filename} -> {name}.png")
            
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
        
        # # 跳过png文件
        # elif ext_lower == '.png':
        #     print(f"跳过png文件: {filename}")

if __name__ == "__main__":
    # 替换为你的文件夹A路径
    folder_a = "D:\\segdino-main\\segdata\\nss\\train\\image"  # 例如："C:/images/folderA" 或 "./folderA"
    convert_jpg_to_png(folder_a)
    print("转换完成")