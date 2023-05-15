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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

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
sum_ = list()
tt = 0
samples = 20
for i, (x, y) in enumerate(train_loader):
    if tt <= samples:
        x = x.reshape((1,28,28))
        x = x.to(device)
        outputs , x_s , c1_s , c2_s , m2_s, h1_s , h2_s , sum_s = model(x)
        _, predicted = outputs.cpu().max(1)
        if predicted==int(y):
            tt+=1
            for j in range(10):
                for m in range(3*5):
                    if m<5:
                        input.append(x_s[j][0].detach().numpy())
                        conv1.append(c1_s[j][0].detach().numpy())
                        conv2.append(c2_s[j][0].detach().numpy())
                        maxpool2.append(m2_s[j][0].detach().numpy()*(5-m)/5)
                        fc1.append(h1_s[j][0].detach().numpy()*m/5)
                        output.append(h2_s[j][0].detach().numpy()*0)
                        sum_.append(sum_s[j][0].detach().numpy())
                    elif m<10:
                        input.append(x_s[j][0].detach().numpy())
                        conv1.append(c1_s[j][0].detach().numpy())
                        conv2.append(c2_s[j][0].detach().numpy())
                        maxpool2.append(m2_s[j][0].detach().numpy()*0)
                        fc1.append(h1_s[j][0].detach().numpy()*(10-m)/5)
                        output.append(h2_s[j][0].detach().numpy()*(m-5)/5)
                        sum_.append(sum_s[j][0].detach().numpy())
                    elif m<15:
                        input.append(x_s[j][0].detach().numpy())
                        conv1.append(c1_s[j][0].detach().numpy())
                        conv2.append(c2_s[j][0].detach().numpy())
                        maxpool2.append(m2_s[j][0].detach().numpy()*0)
                        fc1.append(h1_s[j][0].detach().numpy()*0)
                        output.append(h2_s[j][0].detach().numpy()*(15-m)/5)
                        sum_.append(sum_s[j][0].detach().numpy())
                input.append(x_s[j][0].detach().numpy())
                input.append(x_s[j][0].detach().numpy())
                conv1.append(np.zeros_like(c1_s[j][0].detach().numpy()))
                conv1.append(np.zeros_like(c1_s[j][0].detach().numpy()))
                conv2.append(np.zeros_like(c2_s[j][0].detach().numpy()))
                conv2.append(np.zeros_like(c2_s[j][0].detach().numpy()))
                maxpool2.append(np.zeros_like(m2_s[j][0].detach().numpy()))
                maxpool2.append(np.zeros_like(m2_s[j][0].detach().numpy()))
                fc1.append(np.zeros_like(h1_s[j][0].detach().numpy()))
                fc1.append(np.zeros_like(h1_s[j][0].detach().numpy()))
                output.append(np.zeros_like(h2_s[j][0].detach().numpy()))
                output.append(np.zeros_like(h2_s[j][0].detach().numpy()))
                sum_.append(sum_s[j][0].detach().numpy())
                sum_.append(sum_s[j][0].detach().numpy())

input = np.array(input)
conv1 = np.array(conv1)
conv2 = np.array(conv2)
maxpool2 = np.array(maxpool2)
fc1 = np.array(fc1)
output = np.array(output)
sum_ = np.array(sum_)
np.savez('scnn1_activity.npz', input=input, conv1=conv1, conv2=conv2, maxpool2=maxpool2 ,fc1=fc1, output=output , sum_=sum_)