#预处理后的照片输出
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def resize_keep_aspect(image, max_size):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)

def preprocess(image_path, image_size=512):
    image = Image.open(image_path).convert('RGB')
    image = resize_keep_aspect(image, image_size)  # 保持宽高比，最大边不超过512
    Loader = transforms.Compose([transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
    tensor = Normalize(rgb2bgr(Loader(image) * 255)).unsqueeze(0)
    return tensor

def save_tensor_as_image(tensor, output_path):
    data = tensor.clone().squeeze(0).detach().cpu()
    data = torch.clamp(data, 0, 255) / 255.0  # 将像素值截断到 [0, 255] 再除以 255
    toPIL = transforms.ToPILImage()
    image = toPIL(data)
    image.save(output_path, format='PNG')
    print(f"Saved preprocessed image to: {output_path}")


if __name__ == "__main__":
    content_input = "images/content/c3.jpg"
    style_input = "images/styles/s5.jpg"

    os.makedirs("results/preprocessed_tensor_input", exist_ok=True)

    content_tensor = preprocess(content_input)
    style_tensor = preprocess(style_input)

    save_tensor_as_image(content_tensor, "results/preprocessed_tensor_input/content_tensor_input.png")
    save_tensor_as_image(style_tensor, "results/preprocessed_tensor_input/style_tensor_input.png")
