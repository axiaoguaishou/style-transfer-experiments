#批量运行不同风格权重脚本
import os
import subprocess
import matplotlib.pyplot as plt

# ========================
# 参数配置
# ========================
style_weights = [1e1, 5e1, 1e2, 2e2, 4e2, 8e2]  # 浮点格式
content_image = "images/content/c3.jpg"
style_image = "images/styles/s5.jpg"
init_mode = "image"
init_image = content_image

output_dir = "batch_outputs_style_weight"
log_dir = "loss_logs_style_weight"
script_path = "neural_style.py"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ========================
# 批量运行并记录 loss
# ========================
loss_records = {}

for sw in style_weights:
    sw_tag = str(int(sw))  # 用作文件名标签，例如 "100"
    print(f"\n>>> 正在运行 style_weight = {sw_tag}...")

    out_name = f"output_sw{sw_tag}.png"
    log_name = f"loss_sw{sw_tag}.txt"
    output_image_path = os.path.join(output_dir, out_name)
    log_path = os.path.join(log_dir, log_name)

    command = [
        "python", script_path,
        "-content_image", content_image,
        "-style_image", style_image,
        "-output_image", output_image_path,
        "-style_weight", str(sw),
        "-init", init_mode,
        "-init_image", init_image,
        "-print_iter", "1",
        "-save_iter", "0",
        "-original_colors", "1",
        "-backend", "cudnn", "-cudnn_autotune",
        "-tv_weight", "0"
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
    loss_records[sw_tag] = (iterations, losses)

