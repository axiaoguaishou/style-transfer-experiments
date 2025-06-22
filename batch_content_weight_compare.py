#批量运行不同内容权重脚本

import os
import subprocess
import matplotlib.pyplot as plt

# ========================
# 参数配置
# ========================
content_weights = [1e0, 5e0, 5e1, 1e2, 2e2, 4e2]  # 浮点格式
content_image = "images/content/c3.jpg"
style_image = "images/styles/s5.jpg"
init_mode = "image"
init_image = content_image

output_dir = "batch_outputs_content_weight"
log_dir = "loss_logs_content_weight"
script_path = "neural_style.py"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ========================
# 批量运行并记录 loss
# ========================
loss_records = {}

for cw in content_weights:
    cw_tag = str(int(cw))  # 用作文件名标签，例如 "100"
    print(f"\n>>> 正在运行 content_weight = {cw_tag}...")

    out_name = f"output_cw{cw_tag}.png"
    log_name = f"loss_cw{cw_tag}.txt"
    output_image_path = os.path.join(output_dir, out_name)
    log_path = os.path.join(log_dir, log_name)

    command = [
        "python", script_path,
        "-content_image", content_image,
        "-style_image", style_image,
        "-output_image", output_image_path,
        "-content_weight", str(cw),
        "-style_weight", str(2e2),
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
    loss_records[cw_tag] = (iterations, losses)

