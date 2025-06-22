#批量运行不同CNN模型脚本
import os
import subprocess
import matplotlib.pyplot as plt

# ========================
# 参数配置
# ========================
model_files = {
    "vgg19": "models/vgg19-d01eb7cb.pth",
    "vgg16": "models/vgg16-00b39a1b.pth",
    "nin": "models/nin_imagenet.pth"
}

content_image = "images/content/c3.jpg"
style_image = "images/styles/s5.jpg"
init_mode = "image"
init_image = content_image
style_weight = 2e2

output_dir = "batch_outputs_model"
log_dir = "loss_logs_model"
script_path = "neural_style.py"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ========================
# 批量运行并记录 loss
# ========================
loss_records = {}

for model_name, model_path in model_files.items():
    print(f"\n>>> 正在运行模型: {model_name}...")

    out_name = f"output_{model_name}.png"
    log_name = f"loss_{model_name}.txt"
    output_image_path = os.path.join(output_dir, out_name)
    log_path = os.path.join(log_dir, log_name)

    # 根据模型指定合适的层名
    if model_name == "nin":
        content_layers = "relu0,relu3,relu7,relu12"
        style_layers = "relu0,relu3,relu7,relu12"
    else:
        content_layers = "relu4_2"
        style_layers = "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"

    command = [
        "python", script_path,
        "-content_image", content_image,
        "-style_image", style_image,
        "-output_image", output_image_path,
        "-style_weight", str(style_weight),
        "-init", init_mode,
        "-init_image", init_image,
        "-model_file", model_path,
        "-print_iter", "1",
        "-save_iter", "0",
        "-num_iterations", "1000",
        "-original_colors", "1",
        "-backend", "cudnn", "-cudnn_autotune",
        "-tv_weight", "0",
        "-content_layers", content_layers,
        "-style_layers", style_layers
    ]

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(command, stdout=logfile, stderr=logfile)
        process.wait()

    # 读取 loss 日志
    iterations = []
    losses = []
    with open(log_path, "r") as f:
        for line in f:
            if "Total loss:" in line:
                parts = line.strip().split("Total loss:")
                if len(parts) == 2:
                    try:
                        iterations.append(len(iterations) + 1)
                        losses.append(float(parts[1].strip()))
                    except:
                        continue
    loss_records[model_name] = (iterations, losses)
