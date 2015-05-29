import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.ndimage.filters import median_filter as mf

img_ml=mf(1.0*(255-nd.imread('images/ml.jpg')[::2,::2,::-1]),[8,2,1]).clip(0,255)
fig1, ax1 = plt.subplots(num=1,figsize=(6,1.0*img_ml.shape[0]/img_ml.shape[1]*6))
fig1.subplots_adjust(0,0,1,1);ax1.grid('off')
ax1.axis('off')
fig1.canvas.set_window_title('modified Mona Lisa')
im1 = ax1.imshow(img_ml.astype(np.uint8))