# 国内镜像下载（无警告版）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 下载RSCC数据集（默认自动断点续传）
snapshot_download(
    repo_id="BiliSakura/RSCC",
    local_dir="./data",
    repo_type="dataset"
)

print("✅ 数据集下载完成！存储在 ./data 文件夹")