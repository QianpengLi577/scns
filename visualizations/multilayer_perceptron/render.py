import numpy as np
from mayavi import mlab
import torch
from tqdm import tqdm

from net import Net

model = Net()
model.load_state_dict(torch.load('mnist.pth'))
model.eval()
activity = np.load('activity.npz')

fig = mlab.figure(bgcolor=(13 / 255, 21 / 255, 44 / 255), size=(1920, 1080))
#fig = mlab.figure(bgcolor=(0.05, 0.05, 0.05))

# Layer units
in_z, in_x = np.indices((28, 28))
fc1_z, fc1_x = np.indices((32, 32))
fc2_z, fc2_x = np.indices((32, 32))
fc3_z, fc3_x = np.indices((32, 32))
in_x = in_x.ravel() - 12
in_z = in_z.ravel() - 12
in_y = np.zeros_like(in_x)
fc1_x = fc1_x.ravel() - 16
fc1_z = fc1_z.ravel() - 16
fc1_y = np.ones_like(fc1_x) + 10
fc2_x = fc2_x.ravel() - 16
fc2_z = fc2_z.ravel() - 16
fc2_y = np.ones_like(fc2_x) + 30
fc3_x = fc3_x.ravel() - 16
fc3_z = fc3_z.ravel() - 16
fc3_y = np.ones_like(fc3_x) + 50
out_x = np.arange(10)
out_x = out_x.ravel() - 5
out_y = np.ones_like(out_x) + 80
out_z = np.zeros_like(out_x)

fc1_x = fc1_x + np.random.rand(len(fc1_x)) * 1
fc1_z = fc1_z + np.random.rand(len(fc1_z)) * 1
fc1_y = fc1_y + np.random.rand(len(fc1_y)) * 10
fc2_x = fc2_x + np.random.rand(len(fc2_x)) * 1
fc2_z = fc2_z + np.random.rand(len(fc2_z)) * 1
fc2_y = fc2_y + np.random.rand(len(fc2_y)) * 10
fc3_x = fc3_x + np.random.rand(len(fc3_x)) * 1
fc3_z = fc3_z + np.random.rand(len(fc3_z)) * 1
fc3_y = fc3_y + np.random.rand(len(fc3_y)) * 10
out_x = out_x * 3

# Connections between layers
fc1 = model.fc1.weight.detach().numpy().T
fc2 = model.fc2.weight.detach().numpy().T
fc3 = model.fc3.weight.detach().numpy().T
out = model.fc4.weight.detach().numpy().T
fr_in, to_fc1 = (np.abs(fc1) > 0.1).nonzero()
fr_fc1, to_fc2 = (np.abs(fc2) > 0.05).nonzero()
fr_fc2, to_fc3 = (np.abs(fc3) > 0.05).nonzero()
fr_fc3, to_out = (np.abs(out) > 0.1).nonzero()
fr_fc1 += len(in_x)
to_fc1 += len(in_x)
fr_fc2 += len(in_x) + len(fc1_x)
to_fc2 += len(in_x) + len(fc1_x)
fr_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_out += len(in_x) + len(fc1_x) + len(fc2_x) + len(fc3_x)

# Create the points
x = np.hstack((in_x, fc1_x, fc2_x, fc3_x, out_x))
y = np.hstack((in_y, fc1_y, fc2_y, fc3_y, out_y))
z = np.hstack((in_z, fc1_z, fc2_z, fc3_z, out_z))

act_input = activity['input'][0]
act_fc1 = activity['fc1'][0]
act_fc2 = activity['fc2'][0]
act_fc3 = activity['fc3'][0]
act_out = activity['fc4'][0]
s = np.hstack((
    act_input.ravel() / act_input.max(),
    act_fc1 / act_fc1.max(),
    act_fc2 / act_fc2.max(),
    act_fc3 / act_fc3.max(),
    act_out / act_out.max(),
))

# Layer activation
acts = mlab.points3d(x, y, -z, s, mode='cube', scale_factor=0.5, scale_mode='none', colormap='gray')

# Connections
src = mlab.pipeline.scalar_scatter(x, y, -z, s)
src.mlab_source.dataset.lines = np.vstack((
    np.hstack((fr_in, fr_fc1, fr_fc2, fr_fc3)),
    np.hstack((to_fc1, to_fc2, to_fc3, to_out)),
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

# Text
mlab.text3d(x=-14.5, y=80.5, z=-2, text='0')
mlab.text3d(x=-11.5, y=80.5, z=-2, text='1')
mlab.text3d(x=-8.5, y=80.5, z=-2, text='2')
mlab.text3d(x=-5.5, y=80.5, z=-2, text='3')
mlab.text3d(x=-2.5, y=80.5, z=-2, text='4')
mlab.text3d(x=0.5, y=80.5, z=-2, text='5')
mlab.text3d(x=3.5, y=80.5, z=-2, text='6')
mlab.text3d(x=6.5, y=80.5, z=-2, text='7')
mlab.text3d(x=9.5, y=80.5, z=-2, text='8')
mlab.text3d(x=12.5, y=80.5, z=-2, text='9')

t = mlab.text(0.01, 0.84, 'Type: ML perceptron\nData set: MNIST\nHidden layers: 3\nHidden neurons: 3x1024\nSynapses: 2910992\nSynapses shown: 1%', width=0.15)

mlab.view(azimuth=0, elevation=80, distance=100, focalpoint=[0, 35, 0], reset_roll=False)

# Update the data and view
# @mlab.animate(delay=83, ui=False)
# def anim():
for frame in tqdm(list(range(1600))):
    if frame % 16 == 0:
        i = frame // 16
        act_input = activity['input'][i]
        act_fc1 = activity['fc1'][i]
        act_fc2 = activity['fc2'][i]
        act_fc3 = activity['fc3'][i]
        act_out = activity['fc4'][i]
        s = np.hstack((
            act_input.ravel() / act_input.max(),
            act_fc1 / act_fc1.max(),
            act_fc2 / act_fc2.max(),
            act_fc3 / act_fc3.max(),
            act_out / act_out.max(),
        ))
        acts.mlab_source.scalars = s
        connections.mlab_source.scalars = s
    mlab.savefig('./pic/frame'+str(frame)+'.png')
    print('frame'+str(frame)+'.png')
    mlab.view(azimuth=(frame / 2) % 360, elevation=80, distance=120, focalpoint=[0, 35, 0], reset_roll=False)
        
        # yield

# anim()
