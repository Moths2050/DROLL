import numpy as np
import struct
import os

def NPYReader(filepath):
    img = np.load(filepath)
    col, row = img.shape
    return img, col, row