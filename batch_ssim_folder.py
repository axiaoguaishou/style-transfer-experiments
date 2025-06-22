#批量计算迁移图像SSIM指标脚本
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import argparse

def resize_image_to_match(img, target_size):
    """将图像 img 等比缩放并填充为 target_size 尺寸"""
    w, h = img.size
    target_w, target_h = target_size
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized_img = img.resize((new_w, new_h), Image.LANCZOS)    
    return resized_img.resize(target_size, Image.LANCZOS)

def compute_ssim(img1, img2):
    """计算 SSIM 值（输入均为灰度 PIL 图像）"""
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    score, _ = ssim(arr1, arr2, full=True)
    return round(score, 4)

def process_folder(img1_path, folder_path):
    img1 = Image.open(img1_path).convert('L')  # 转灰度
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img2_path = os.path.join(folder_path, filename)
            try:
                img2 = Image.open(img2_path).convert('L')
                resized_img1 = resize_image_to_match(img1, img2.size)
                score = compute_ssim(resized_img1, img2)
                results.append((filename, score))
                print(f"{filename} => SSIM: {score}")
            except Exception as e:
                print(f"❌ 跳过 {filename}：{e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量计算某文件夹下图片与固定图片的 SSIM 值")
    parser.add_argument("fixed_image", type=str, help="固定图像路径（如内容图）")
    parser.add_argument("folder", type=str, help="待比较图像所在的文件夹路径")
    args = parser.parse_args()

    process_folder(args.fixed_image, args.folder)
