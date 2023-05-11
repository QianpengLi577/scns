from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
from spiking_model_sample_one import SCNN1
device = torch.device("cpu")

data_path =  './data/' #todo: input your data path
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())

aa= torch.load("scnn_mnist.pth",map_location=torch.device('cpu'))
import numpy as np
# snn.eval().to('cpu')
model = SCNN1()
model.to(device)
model.fc1.weight.data = aa['fc1.weight'].data.cpu()
model.fc1.bias.data = aa['fc1.bias'].data.cpu()
model.fc2.weight.data = aa['fc2.weight'].data.cpu()
model.fc2.bias.data = aa['fc2.bias'].data.cpu()
model.conv1.weight.data = aa['conv1.weight'].data.cpu()
model.conv1.bias.data = aa['conv1.bias'].data.cpu()
model.conv2.weight.data = aa['conv2.weight'].data.cpu()
model.conv2.bias.data = aa['conv2.bias'].data.cpu()

input = list()
conv1 = list()
conv2 = list()
maxpool2 = list()
fc1 = list()
output = list()
for i in range(20):
    x , y = train_dataset[i]
    x = x.to(device)
    outputs , x_s , c1_s , c2_s , m2_s, h1_s , h2_s = model(x)
    for j in range(10):
        input.append(x_s[j][0].detach().numpy())
        input.append(x_s[j][0].detach().numpy())
        conv1.append(c1_s[j][0].detach().numpy())
        conv1.append(np.zeros_like(c1_s[j][0].detach().numpy()))
        conv2.append(c2_s[j][0].detach().numpy())
        conv2.append(np.zeros_like(c2_s[j][0].detach().numpy()))
        maxpool2.append(m2_s[j][0].detach().numpy())
        maxpool2.append(np.zeros_like(m2_s[j][0].detach().numpy()))
        fc1.append(h1_s[j][0].detach().numpy())
        fc1.append(np.zeros_like(h1_s[j][0].detach().numpy()))
        output.append(h2_s[j][0].detach().numpy())
        output.append(np.zeros_like(h2_s[j][0].detach().numpy()))

input = np.array(input)
conv1 = np.array(conv1)
conv2 = np.array(conv2)
maxpool2 = np.array(maxpool2)
fc1 = np.array(fc1)
output = np.array(output)
np.savez('scnn_activity.npz', input=input, conv1=conv1, conv2=conv2, maxpool2=maxpool2 ,fc1=fc1, output=output)