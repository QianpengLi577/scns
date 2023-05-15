import numpy as np
from mayavi import mlab
import torch
from tqdm import tqdm

# from net import Net
from spiking_model import*

model = SCNN()
model.load_state_dict(torch.load('scnn_mnist.pth',map_location=torch.device('cpu')))
model.eval()
activity = np.load('scnn1_activity.npz')
act_input = activity['input'][1]
act_conv1 = activity['conv1'][1]
act_conv2 = activity['conv2'][1]
act_maxpool2 = activity['maxpool2'][1]
act_fc1 = activity['fc1'][1]
act_out = activity['output'][1]
sum_ = activity['sum_'][1]

fig = mlab.figure(bgcolor=(13 / 255, 21 / 255, 44 / 255), size=(1920, 1080))

img_input = mlab.imshow(act_input, colormap='gray', interpolate=False, extent=[-27, 27, -27, 27, 1, 1])
img_input.actor.position = [0, 0, 0]
img_input.actor.orientation = [0, 90, 90]
img_input.actor.force_opaque = True

tange1 = [i for i in range(30,150,1)]
tange2 = [180-i for i in range(30,150,1)]
aa = np.array(np.hstack((np.array(tange1),np.array(tange2))))

# Conv1
img_conv1 = list()
for row in range(2):
    for col in range(2):
        img = mlab.imshow(act_conv1[row * 2 + col], colormap='gray', interpolate=False)
        img.actor.position = [(row - 0.5) * 28, 40, (col - 0.5) * 28]
        img.actor.orientation = [0, 90, 90]
        img.actor.force_opaque = True
        img_conv1.append(img)

# Conv2
img_conv2 = list()
for row in range(2):
    for col in range(2):
        img = mlab.imshow(act_conv2[row * 2 + col], colormap='gray', interpolate=False)
        img.actor.position = [(row - 0.5) * 14, 80, (col - 0.5) * 14]
        img.actor.orientation = [0, 90, 90]
        img.actor.force_opaque = True
        img_conv2.append(img)

# # maxpool2
# img_maxpool2 = list()
# for row in range(2):
#     for col in range(2):
#         img = mlab.imshow(act_maxpool2[row * 2 + col], colormap='gray', interpolate=False)
#         img.actor.position = [(row - 0.5) * 7, 120, (col - 0.5) * 7]
#         img.actor.orientation = [0, 90, 90]
#         img.actor.force_opaque = True
#         img_maxpool2.append(img)

# Create the points
max2_x = list()
max2_y = list()
for row in range(2):
    for col in range(2):
        x, y = np.indices((7, 7))
        x = x.ravel() + (row - 1) * 7
        y = y.ravel() + (col - 1) * 7
        max2_x.append(x)
        max2_y.append(y)
max2_x = np.hstack(max2_x)
max2_y = np.hstack(max2_y)
max2_z = np.ones_like(max2_x) * 120

fc1_x, fc1_y = np.indices((12, 12))
fc1_x = fc1_x.ravel() * 2 - 12
fc1_y = fc1_y.ravel() * 2 - 12
fc1_z = np.ones_like(fc1_x) * 160 + np.random.rand(*fc1_x.shape) * 10

out_x = np.arange(10)
out_x = (out_x.ravel() - 5) * 3
out_y = np.zeros_like(out_x)
out_z = np.ones_like(out_y) + 200

sum_x = np.arange(10)
sum_x = (sum_x.ravel() - 5) * 3
sum_y = np.zeros_like(sum_x)
sum_z = np.ones_like(sum_y) + 230

x = np.hstack([max2_x, fc1_x, out_y , sum_y])
y = np.hstack([max2_y, fc1_y, out_x , sum_x])
z = np.hstack([max2_z, fc1_z, out_z , sum_z])
s = np.hstack([
    act_maxpool2.ravel() ,
    act_fc1 ,
    act_out,
    sum_ / (sum_.max()+0.000000001),
])
# acts = mlab.points3d(x[len(act_maxpool2.ravel()):], z[len(act_maxpool2.ravel()):], y[len(act_maxpool2.ravel()):], s[len(act_maxpool2.ravel()):], mode='cube', scale_factor=1, scale_mode='none', colormap='gray')
acts = mlab.points3d(x, z, y, s, mode='cube', scale_factor=1, scale_mode='none', colormap='gray')

# Connections between the layers
fc1 = model.fc1.weight.detach().numpy().T
out = model.fc2.weight.detach().numpy().T
fr_conv2, to_fc1 = (np.abs(fc1) > 0.1).nonzero()
fr_fc1, to_out = (np.abs(out) > 0.2).nonzero()
fr_out = np.arange(10)
to_sum = np.arange(10)

to_fc1 += len(max2_x)
fr_fc1 += len(max2_x)
to_out += len(max2_x) + len(fc1_x)
fr_out += len(max2_x) + len(fc1_x)
to_sum += len(max2_x) + len(fc1_x) + len(out_x)
c = np.vstack((
    np.hstack((fr_conv2, fr_fc1 , fr_out)),
    np.hstack((to_fc1, to_out , to_sum)),
)).T

src = mlab.pipeline.scalar_scatter(x, z, y, s)
src.mlab_source.dataset.lines = np.vstack((
    np.hstack((fr_conv2, fr_fc1 , fr_out)),
    np.hstack((to_fc1, to_out , to_sum)),
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

# Text
mlab.text3d(z=-15, y=232, x=-2, text='0')
mlab.text3d(z=-12, y=232, x=-2, text='1')
mlab.text3d(z=-9,  y=232, x=-2, text='2')
mlab.text3d(z=-6,  y=232, x=-2, text='3')
mlab.text3d(z=-3,  y=232, x=-2, text='4')
mlab.text3d(z=0,   y=232, x=-2, text='5')
mlab.text3d(z=3,   y=232, x=-2, text='6')
mlab.text3d(z=6,   y=232, x=-2, text='7')
mlab.text3d(z=9,   y=232, x=-2, text='8')
mlab.text3d(z=12,  y=232, x=-2, text='9')

mlab.view(azimuth=0,elevation= 90, distance=300 ,focalpoint= [0, 90, 0])

# Update the data and view
# @mlab.animate(delay=83, ui=True)
# def anim():
for frame in tqdm(list(range(20*10*2*4))):
    if frame % 4 == 0:
        i = frame // 4
        img_input.mlab_source.scalars = activity['input'][i]
        for img, a in zip(img_conv1, activity['conv1'][i]):
            img.mlab_source.scalars = a
        for img, a in zip(img_conv2, activity['conv2'][i]):
            img.mlab_source.scalars = a
        # for img, a in zip(img_maxpool2, activity['maxpool2'][i]):
        #     img.mlab_source.scalars = a
        act_fc1 = activity['fc1'][i]
        act_out = activity['output'][i]
        sum_ = activity['sum_'][i]
        s = np.hstack([
            activity['maxpool2'][i].ravel() ,
            act_fc1 ,
            act_out,
            sum_ / (sum_.max()+0.000000001),
            ])
        # acts.mlab_source.scalars = s[len(act_maxpool2.ravel()):]
        acts.mlab_source.scalars = s
        connections.mlab_source.scalars = s
    
    mlab.view(azimuth=aa[int(frame%aa.shape[0])], elevation=75, distance=350, focalpoint=[0, 90, 0], reset_roll=False)
    mlab.savefig('pic1\\frame'+str(frame)+'.png',figure=fig)    
        # yield

# anim()
