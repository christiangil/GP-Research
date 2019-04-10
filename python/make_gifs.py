from PIL import Image
import numpy as np
import glob, os

directory_with_images = "C:/Users/Christian/Dropbox/GP_research/julia/figs/gp/quasi_periodic_kernel/training/"
image_type = ".png"

os.chdir(directory_with_images)

for j in ["gp","gp_K_profile"]:

    for i in range(1,4):

        filenames = "*" + j + "_%d"%i

        # open all of the images
        images=[]
        for file in glob.glob(filenames + image_type):
            images.append(Image.open(file))
            print(file)

        # combine the images into a gif
        images[0].save(j + '%d.gif'%i,
            save_all=True,
            append_images=images[1:],
            duration=300,
            loop=0)
