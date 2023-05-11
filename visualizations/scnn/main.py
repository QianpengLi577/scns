# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
from spiking_model_sample_one import SCNN1
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
names = 'spiking_model'
data_path =  './data/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = snn(images)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)

torch.save(snn.state_dict(), "scnn_mnist.pth")

##
import numpy as np
snn.eval().to('cpu')
model = SCNN1()
model.fc1.weight.data = snn.fc1.weight.data
model.fc1.bias.data = snn.fc1.bias.data
model.fc2.weight.data = snn.fc2.weight.data
model.fc2.bias.data = snn.fc2.bias.data
model.conv1.weight.data = snn.conv1.weight.data
model.conv1.bias.data = snn.conv1.bias.data
model.conv2.weight.data = snn.conv2.weight.data
model.conv2.bias.data = snn.conv2.bias.data
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