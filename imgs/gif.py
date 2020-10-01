import imageio
import os 
import numpy as np 
from natsort import natsorted
images = natsorted([a for a in os.listdir(".") if ".jpg" in a and ".gif" not in a ])

for filename in images:
	print(filename)
	img = np.array(imageio.imread(filename))
	print(img.shape)
	images.append(img)


images = np.array(images)
print(images.shape)

print("Loaded. ")

imageio.mimsave('animation.gif', np.array(images))
