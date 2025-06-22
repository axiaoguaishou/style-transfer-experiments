#绘制不同CNN模型的联合损失函数
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ========================
# 设置日志路径和初始化标签
# ========================
log_dir = "logs/loss_logs_model"  # 日志文件夹
log_files = {
    "VGG-19": "loss_vgg19.txt",
    "VGG-16": "loss_vgg16.txt",
    "NIN": "loss_nin.txt"  
}

colors = {
    "VGG-19": "red",
    "VGG-16": "green",
    "NIN": "blue"
}

# ========================
# 读取每个日志中的 loss 数据
# ========================
loss_records = {}

for label, filename in log_files.items():
    filepath = os.path.join(log_dir, filename)
    iterations = []
    losses = []
    with open(filepath, "r") as f:
        for line in f:
            if "Total loss:" in line:
                parts = line.strip().split("Total loss:")
                if len(parts) == 2:
                    try:
                        iterations.append(len(iterations) + 1)
                        losses.append(float(parts[1].strip()))
                    except:
                        continue
    loss_records[label] = (iterations, losses)

# ========================
# 绘制图像（对数纵坐标）
# ========================
plt.figure(figsize=(10, 6))
for label, (iters, losses) in loss_records.items():
    plt.plot(iters, losses, label=f"{label}", color=colors[label], linewidth=2)

plt.xlabel("Iteration", fontsize=16, fontname='Times New Roman')
plt.ylabel("Total Loss", fontsize=16, fontname='Times New Roman')
plt.title("Total Loss for Different Models", fontsize=18, fontname='Times New Roman')
plt.yscale("log")
plt.xlim(left=0)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(
    prop=font_manager.FontProperties(
        family='Times New Roman',  # 字体
        size=16,                   # 字体大小
        weight='normal'            # 字重，可选 'bold'
    ),
    loc='upper right'
)

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("results/batch_outputs_model/model_loss_log_curve.png")
plt.show()