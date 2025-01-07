import numpy as np
import scipy.signal

def SLFilter(inum, det_spacing):
    det2 = det_spacing*det_spacing
    pi2 =np.pi * np.pi

    ic = np.floor(inum/2)
    sl_filter = np.zeros([1, inum], dtype=np.float)

    for i in range(inum):
        fpos = i - ic
        sl_filter[0, i] = -1.0 / (pi2 * det2 * (4 * fpos * fpos - 1.0))


    return sl_filter

def SLFiltering(image, sl_filter):
    cols = image.shape[1]
    rows = image.shape[0]

    ilen = sl_filter.shape[1]
    tot_len = cols + ilen - 1

    ic    = np.int(tot_len/2)
    ihalf = np.int(cols/2)

    Res = np.zeros([rows, tot_len], dtype=np.float32)

    for i in range(rows):
        Res[i, :] = scipy.signal.convolve(image[i], sl_filter[0])

    image[1:rows, 1:cols] = Res[1:rows, (ic - ihalf + 1):(ic + ihalf)]

    return image