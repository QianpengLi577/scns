import numpy as np
from mayavi import mlab
import torch
from tqdm import tqdm

# from net import Net
# from spiking_model import*

'''
color map
accent flag hot pubu set2
autumn gist_earth hsv pubugn set3
black-white gist_gray jet puor spectral
blue-red gist_heat oranges purd spring
blues gist_ncar orrd purples summer
bone gist_rainbow paired rdbu winter
brbg gist_stern pastel1 rdgy ylgnbu
bugn gist_yarg pastel2 rdpu ylgn
bupu gnbu pink rdylbu ylorbr
cool gray piyg rdylgn ylorrd
copper greens prgn reds
dark2 greys prism set1
'''


# fig = mlab.figure(bgcolor=(72 / 255, 61 / 255, 139 / 255), size=(1920, 1080))
fig = mlab.figure(bgcolor=(9 / 255, 39 / 255, 59 / 255), size=(1920, 1080))
# fig = mlab.figure(size=(1920, 1080))
r1 = 4
c1 = 4

data = (np.random.rand(100,r1*c1) > 0.8 ).astype(np.int32)
data_ = np.zeros((1000,r1*c1))
for i in range(100):
    for j in range(10):
        data_[i*10+j,:] = data[i,:]*(1-j/10)

# Create the points
max2_x = list()
max2_y = list()
for row in range(r1):
    for col in range(c1):
        x, y = np.indices((1, 1))
        x = x.ravel() - (row - 1) * 7
        y = y.ravel() - (col - 1) * 7
        max2_x.append(x)
        max2_y.append(y)
max2_x = np.hstack(max2_x)
max2_y = np.hstack(max2_y)
max2_z = np.ones_like(max2_x) * 50


x = np.hstack([ max2_y])
y = np.hstack([max2_x])
z = np.hstack([max2_z])
s = np.hstack([
    data_[0,:],
])

# acts = mlab.points3d(x[len(act_maxpool2.ravel()):], z[len(act_maxpool2.ravel()):], y[len(act_maxpool2.ravel()):], s[len(act_maxpool2.ravel()):], mode='cube', scale_factor=1, scale_mode='none', colormap='gray')
acts1 = mlab.points3d(x, z, y, s, mode='cube', scale_factor=3, scale_mode='none', colormap='gray', vmin=0 ,vmax =1)

# Connections between the layers
fc = np.zeros((r1*c1,c1*r1))
for i in range (r1):
    for j in range (c1):
        if i==r1-1:
            if j!=c1-1:
                fc[i*c1+j][i*c1+j+1]=1
        if i!=r1-1:
            if j!=c1-1:
                fc[i*c1+j][i*c1+j+1]=1
            fc[i*c1+j][(i+1)*c1+j]=1
fr_conv2, to_fc1 = (np.abs(fc) > 0.2).nonzero()

c = np.vstack((
    fr_conv2,
    to_fc1,
)).T

src = mlab.pipeline.scalar_scatter(x, z, y, s)
src.mlab_source.dataset.lines = np.vstack((
    fr_conv2,
    to_fc1,
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=4.0, opacity=0.5)

t = mlab.text(0.25, 0.54, ' ', width=0.15)

mlab.view(azimuth=0,elevation= 90, distance=340 ,focalpoint= [0, 90, 0])

# Update the data and view
# @mlab.animate(delay=83, ui=True)
# def anim():
for frame in tqdm(list(range(100*10))):
    if frame % 1 == 0:
        i = frame // 1
        s = np.hstack([
            data_[i,:],
            ])
        acts1.mlab_source.scalars = s
        connections.mlab_source.scalars = s
    
    # mlab.view(azimuth=aa[int((frame/4)%aa.shape[0])], elevation=75, distance=400, focalpoint=[0, 90, 0], reset_roll=False)
    if data[i//10,:].max()==0:
        str_print = 'NO CORE FIRED'
    else:
        y = (data[i//10,:] > 0.2).nonzero()
        str_print = 'FIRED CORE ID IS '
        for h in range(y[0].shape[0]):
            AAAA = '\nROW : ' + str(y[0][h]//c1) + ' COL : ' + str(y[0][h]%c1)
            str_print += AAAA
    t.text = str_print
    mlab.view(azimuth=90, elevation=75, distance=50, focalpoint=[0, 90, 0], reset_roll=False)
    mlab.savefig('pic4\\frame'+str(frame)+'.png',figure=fig)    
        # yield

# anim()
