import torch
import os
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url

project_dir = r"F:\学习\人工智能\课程论文\代码\neural-style-pt"
models_dir = os.path.join(project_dir, "models")

# Download the VGG-19 model and fix the layer names
print("Downloading the VGG-19 model")
sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, os.path.join(models_dir, "vgg19-d01eb7cb.pth"))

# Download the VGG-16 model and fix the layer names
print("Downloading the VGG-16 model")
sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, os.path.join(models_dir, "vgg16-00b39a1b.pth"))

# Download the NIN model
print("Downloading the NIN model")
sd = load_url("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth")
torch.save(sd, os.path.join(models_dir, "nin_imagenet.pth"))

print("All models have been successfully downloaded")
