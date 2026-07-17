from huggingface_hub import snapshot_download
import os


download_path = "D:\\segdino-main\\web_pth"
token = "your_token"

print("开始下载.....")
snapshot_download(repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
                  local_dir=download_path,
                  local_dir_use_symlinks=False,
                  token=token)

print("下载完成!")
