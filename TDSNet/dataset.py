import os
from typing import List, Tuple, Optional
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

# --- ImageNet mean/std ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class ResizeAndNormalize:
    def __init__(self, size=(256,256), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):   # ✅ 参数改名，接收 numpy 数组
        # --- image ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_CUBIC)
        img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img_t = TF.normalize(img_t, self.mean, self.std)

        # --- mask ---
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 1
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)

        return img_t, mask_t

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def _list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    out.sort()
    return out

def _swap_dir_keep_name(path: str, src_dir_name: str, dst_dir_name: str, dst_ext: Optional[str]) -> str:
    parts = path.replace("\\", "/").split("/")
    i = parts.index(src_dir_name)
    parts[i] = dst_dir_name
    dst_path = "/".join(parts)
    if dst_ext is not None:
        base, _ = os.path.splitext(dst_path)
        dst_path = base + dst_ext
    return dst_path

class FolderDataset(data.Dataset):
    def __init__(self, root, split="train", img_dir_name="img", label_dir_name="label",
                 img_exts=SUPPORTED_EXTS, mask_ext=None, transform=None, strict_pair=True):
        super().__init__()
        self.img_dir = os.path.join(root, split, img_dir_name)
        self.label_dir = os.path.join(root, split, label_dir_name)
        self.transform = transform
        self.mask_ext = mask_ext
        self.strict_pair = strict_pair

        self.img_paths = _list_files(self.img_dir, img_exts)
        self.pairs = []
        for ip in self.img_paths:
            mp = _swap_dir_keep_name(ip, img_dir_name, label_dir_name, mask_ext)
            if not os.path.isfile(mp):
                if strict_pair:
                    raise FileNotFoundError(f"Mask not found for image {ip}")
                else:
                    continue
            self.pairs.append((ip, mp))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if img is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        img_t, mask_t = self.transform(img, mask)
        
        # 1. 提取原文件名（从图像路径中获取）
        # 示例：img_path 为 "data/train/image/abc.jpg" → 提取 "abc.jpg"（带扩展名）
        img_name = os.path.basename(img_path)
        # 若需要去除扩展名（如 "abc.jpg" → "abc"），则用下面这行替换上面一行：
        # img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 2. （可选）保留原 meta 信息（如需）
        meta = {"img_path": img_path, "mask_path": mask_path, "img_name": img_name}
        
        # 3. 修改返回值：返回 (图像张量, 标签张量, 原文件名)
        # 注意：这里用 img_name 替代原来的 meta，以便在评估时直接获取文件名
        return img_t, mask_t, img_name

