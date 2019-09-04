# neural-style-pt Installation

This guide will walk you through the setup for `neural-style-pt` on Ubuntu. If you wish to install PyTorch on a different operating system like Windows or MacOS, installation guides can be found [here](https://pytorch.org). 

Note that in order to reduce their size, the pre-packaged binary releases (pip, Conda, etc...) have removed support for some older GPUs, and thus you will have to install from source in order to use these GPUs. 


## With A Package Manager:

The pip and Conda packages ship with CUDA and cuDNN already built in, so after you have installed PyTorch with pip or Conda, you can skip to [installing neural-style-pt](https://github.com/ProGamerGov/neural-style-pt/blob/master/INSTALL.md#install-neural-style-pt). 

### pip:

If you wish to install neural-style-pt as a pip package, then use the following command: 

```
# in a terminal, run the command
pip install neural-style
```

Or:


```
# in a terminal, run the command
pip3 install neural-style
```

Next download the models with: 


```
neural-style -download_models
```

By default the models are downloaded to your home directory, but you can specify a download location with: 

```
neural-style -download_models <download_path>
```

#### Github and pip:

Following the pip installation instructions
[here](http://pytorch.org), you can install PyTorch with the following commands:

```
# in a terminal, run the commands
cd ~/
pip install torch 
pip install torchvision 
```

Or:

```
cd ~/
pip3 install torch 
pip3 install torchvision 
```

### Conda:

Following the Conda installation instructions
[here](http://pytorch.org), you can install PyTorch with the following command:

```
conda install pytorch torchvision -c pytorch
```


## From Source:

### (Optional) Step 1: Install CUDA

If you have a [CUDA-capable GPU from NVIDIA](https://developer.nvidia.com/cuda-gpus) then you can
speed up `neural-style-pt` with CUDA. 

First download and unpack the local CUDA installer from NVIDIA; note that there are different
installers for each recent version of Ubuntu:

```
# For Ubuntu 17.0.4
sudo dpkg -i cuda-repo-ubuntu1704-9-1-local_9.1.85-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
```

```
# For Ubuntu 16.0.4
sudo dpkg -i cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
```

Instructions for downloading and installing the latest CUDA version on all supported operating systems, can be found [here](https://developer.nvidia.com/cuda-downloads).  

Now update the repository cache and install CUDA. Note that this will also install a graphics driver from NVIDIA.

```
sudo apt-get update
sudo apt-get install cuda
```

At this point you may need to reboot your machine to load the new graphics driver.
After rebooting, you should be able to see the status of your graphics card(s) by running
the command `nvidia-smi`; it should give output that looks something like this:

```
Wed Apr 11 21:54:49 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.90                 Driver Version: 384.90                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   62C    P0    68W / 149W |      0MiB / 11439MiB |     94%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### (Optional) Step 2: Install cuDNN

cuDNN is a library from NVIDIA that efficiently implements many of the operations (like convolutions and pooling)
that are commonly used in deep learning.

After registering as a developer with NVIDIA, you can [download cuDNN here](https://developer.nvidia.com/cudnn). Make sure that you use the approprite version of cuDNN for your version of CUDA. 

After dowloading, you can unpack and install cuDNN like this:

```
tar -xvzf cudnn-9.1-linux-x64-v7.1.tgz
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.1/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.1/include/
```

Note that the cuDNN backend can only be used for GPU mode.


### Step 3: Install PyTorch

To install PyTorch [from source](https://github.com/pytorch/pytorch#from-source) on Ubuntu (Instructions may be different if you are using a different OS): 

```
cd ~/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

cd ~/
git clone --recursive https://github.com/pytorch/vision
cd vision
python setup.py install
```

To check that your torch installation is working, run the command `python` or `python3` to enter the Python interpreter. Then type `import torch` and hit enter. 

You can then type `print(torch.version.cuda)` and `print(torch.backends.cudnn.version())` to confirm that you are using the desired versions of CUDA and cuDNN.

To quit just type `exit()` or use  Ctrl-D.


## Install neural-style-pt

First we clone `neural-style-pt` from GitHub:

```
cd ~/
git clone https://github.com/ProGamerGov/neural-style-pt.git
cd neural-style-pt
```

Next we need to download the pretrained neural network models:

```
python models/download_models.py
```

You should now be able to run `neural-style-pt` in CPU mode like this:

```
python neural_style.py -gpu -1 -print_iter 1
```

If you installed PyTorch with support for CUDA, then should now be able to run `neural-style-pt` in GPU mode like this:

```
python neural_style.py -gpu 0 -print_iter 1
```

If you installed PyTorch with support for cuDNN, then you should now be able to run `neural-style-pt` with the `cudnn` backend like this:

```
python neural_style.py -gpu 0 -backend cudnn -print_iter 1
```

If everything is working properly you should see output like this:

```
Iteration 1 / 1000
  Content 1 loss: 1616196.125
  Style 1 loss: 29890.9980469
  Style 2 loss: 658038.625
  Style 3 loss: 145283.671875
  Style 4 loss: 11347409.0
  Style 5 loss: 563.368896484
  Total loss: 13797382.0
Iteration 2 / 1000
  Content 1 loss: 1616195.625
  Style 1 loss: 29890.9980469
  Style 2 loss: 658038.625
  Style 3 loss: 145283.671875
  Style 4 loss: 11347409.0
  Style 5 loss: 563.368896484
  Total loss: 13797382.0
Iteration 3 / 1000
  Content 1 loss: 1579918.25
  Style 1 loss: 29881.3164062
  Style 2 loss: 654351.75
  Style 3 loss: 144214.640625
  Style 4 loss: 11301945.0
  Style 5 loss: 562.733032227
  Total loss: 13711628.0
Iteration 4 / 1000
  Content 1 loss: 1460443.0
  Style 1 loss: 29849.7226562
  Style 2 loss: 643799.1875
  Style 3 loss: 140405.015625
  Style 4 loss: 10940431.0
  Style 5 loss: 553.507446289
  Total loss: 13217080.0
Iteration 5 / 1000
  Content 1 loss: 1298983.625
  Style 1 loss: 29734.8964844
  Style 2 loss: 604133.8125
  Style 3 loss: 125455.945312
  Style 4 loss: 8850759.0
  Style 5 loss: 526.118591309
  Total loss: 10912633.0
```
