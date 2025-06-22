
# 基于 PyTorch 的图像风格迁移复现项目

本项目基于 GitHub 项目 [ProGamerGov/neural-style-pt](https://github.com/ProGamerGov/neural-style-pt) 进行深度复现与扩展，使用 PyTorch 实现图像风格迁移，支持**不同初始化方式**、**多种 CNN 模型（VGG19/VGG16/NIN）**、**不同内容与风格权重参数**组合对比实验，具备**可视化损失曲线**与**SSIM评价**能力，适用于人工智能课程设计与生成式图像理解研究。

##  项目结构说明

```
neural-style-pt/
│
├── neural_style.py                      # 主风格迁移程序
├── CaffeLoader.py                       # 模型加载器
├── batch_init_compare.py                # 初始化方式对比实验
├── batch_style_weight_compare.py        # 风格权重对比实验
├── batch_model_compare.py               # CNN 模型对比实验
├── plot_log_losses_init.py              # 绘制不同初始化方式总损失对比图
├── plot_log_losses_model.py             # 绘制不同 CNN 模型总损失对比图
├── plot_log_losses_content_weight.py    # 绘制不同内容权重总损失对比图
├── plot_log_losses_style_weight.py      # 绘制不同风格权重总损失对比图
├── compute_ssim_folder.py               # 批量计算 SSIM
├── preprocessed_images.py               # 预处理后的图像可视化
│
├── models/                              # CNN 模型目录
├── images/                              # 输入图片（内容图、风格图）
├── logs/                                # 各实验的损失日志
├── results/                             # 输出的风格迁移图像、损失对比图和预处
│                                          理输入图像
└── README.md                            # 本文件
```

## 依赖环境配置
推荐使用 conda 创建独立环境，环境名字可自行更改，以下均以 `nst_gpu` 这个虚拟环境为例，在 Anaconda PowerShell Prompt (Miniconda3) 中逐行运行下面的命令，先是创建一个独立环境，然后激活该环境，在该环境下安装 GPU 版本 PyTorch和其他必须的库
```bash
conda create -n nst_gpu python=3.10
conda activate nst_gpu

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pillow numpy scipy matplotlib
conda install -c conda-forge scikit-image
```

## 模型文件准备

在 `models/` 目录下放置以下的三个模型文件，需要通过运行该目录下的程序 `models\download_models.py` 来下载三个模型文件，可以根据需要只下载其中一个，包含了 VGG-19，VGG-16，NIN，里面的 `project_dir` 需要改成自己存放这个项目的路径。

- `vgg19-d01eb7cb.pth`
- `vgg16-00b39a1b.pth`
- `nin_imagenet.pth`

## 运行示例
推荐使用 Anaconda PowerShell Prompt (Miniconda3) 运行代码，只需先激活之前创立的虚拟环境，再进入项目目录即可运行代码。下面的项目目录需要改成自己的目录，可以根据自己的需要调整初始不同的参数

```bash
conda activate nst_gpu
cd F:\学习\人工智能\课程论文\代码\neural-style-pt

python neural_style.py -content_image images/content/c3.jpg -style_image images/styles/s5.jpg -output_image images/results/r1.png -init image -init_image images/content/c3.jpg -style_weight 2e2 -content_weight 5e0 -tv_weight 0 -original_colors 1 
```

## 批量实验复现

### 初始化方式对比

```bash
python batch_init_compare.py
```

### CNN 模型对比

```bash
python batch_model_compare.py
```
### 内容权重对比

```bash
python batch_content_weight_compare.py
```

### 风格权重对比

```bash
python batch_style_weight_compare.py
```

## 绘制损失函数对比图

```bash
python plot_log_losses_init.py
python plot_log_losses_model.py
python plot_log_losses_content_weight.py
python plot_log_losses_style_weight.py
```

## 批量计算 SSIM
第一个路径为基准照片，第二个路径为所需要计算 SSIM 的所有图片所在的目录，这里以内容图像为第一个路径，以不同初始化生成的图片为第二个路径
```bash
python compute_ssim.py images/content/c3.jpg batch_outputs_init\output_init_image_content.png
```

## 预处理输入图像可视化
```bash
python save_raw_tensor_input.py --input images/content/pic.jpg --output preprocessed_input.png
```

