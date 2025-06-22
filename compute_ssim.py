#计算迁移图像的SSIM指标
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import argparse
import os

def resize_image_max_512(img_path):
    """
    加载图像，并将其等比缩放到最大边不超过512像素
    返回：缩放后的 PIL.Image 对象
    """
    img = Image.open(img_path).convert('L')  # 转灰度
    w, h = img.size
    max_size = max(w, h)
    if max_size > 512:
        scale = 512.0 / max_size
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def compute_ssim(img1_path, img2_path):
    """
    加载两个图像，缩放 img1 到最大边为512，再计算它与 img2 的 SSIM
    两张图尺寸会自动对齐为相同大小
    """
    img1 = resize_image_max_512(img1_path)
    img2 = Image.open(img2_path).convert('L')

    # 将 img1 等比缩放后再缩放/填充为 img2 的尺寸
    target_size = img2.size
    img1 = img1.resize(target_size, Image.LANCZOS)

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    score, _ = ssim(arr1, arr2, full=True)
    return round(score, 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSIM with resized input")
    parser.add_argument("image1", type=str, help="Path to content image (will be resized)")
    parser.add_argument("image2", type=str, help="Path to generated image")
    args = parser.parse_args()

    score = compute_ssim(args.image1, args.image2)
    print(f"\n图像结构相似性 SSIM（保留四位小数）:\n{args.image1} vs {args.image2} => SSIM = {score}\n")
