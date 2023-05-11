from PIL import Image

im = Image.open('pic\\frame0.png')
images = []
for i in range(600-1):
    images.append(Image.open('pic\\frame'+str(i+1)+'.png'))
im.save('scnn_spike.gif',save_all=True,append_images=images,duration=100)