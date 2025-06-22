#批量运行不同初始化方式脚本
import os
import subprocess
import matplotlib.pyplot as plt

# ========================
# 配置参数
# ========================
init_configs = {
    "random": {"init": "random", "init_image": None},
    "image_content": {"init": "image", "init_image": "images/content/c3.jpg"},
    "image_style": {"init": "image", "init_image": "images/styles/s5.jpg"}
}

content_image = "images/content/c3.jpg"
style_image = "images/styles/s5.jpg"
output_dir = "batch_outputs_init"
script_path = "neural_style.py"
log_dir = "loss_logs_init"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ========================
# 批量运行并记录 loss
# ========================
loss_records = {}

for name, config in init_configs.items():
    print(f"\n>>> 正在运行 init = {name}...")
    out_name = f"output_init_{name}.png"
    log_name = f"loss_init_{name}.txt"
    output_image_path = os.path.join(output_dir, out_name)
    log_path = os.path.join(log_dir, log_name)

    command = [
        "python", script_path,
        "-content_image", content_image,
        "-style_image", style_image,
        "-output_image", output_image_path,
        "-style_weight", str(2e2),
        "-init", config["init"],
        "-print_iter", "1",
        "-save_iter", "0",
        "-original_colors", "1",
        "-backend", "cudnn", "-cudnn_autotune",
        "-tv_weight", "0"

    ]

    if config["init_image"]:
        command.extend(["-init_image", config["init_image"]])

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
    loss_records[name] = (iterations, losses)


