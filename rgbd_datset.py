#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 00:06:48 2021

@author: anirudhputrevu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

b = np.load('/Users/anirudhputrevu/Downloads/results/depth_training_data.npz')
print(b.files)

data = b['rgb_images']

display(data)
data.shape

# reshape to 3500*3 , 96*96 

x = np.reshape(data,(10500,9216))

display(x)
x.shape

plt.imshow(x[1].reshape((96,96)),cmap=plt.cm.spring)
plt.show()
