from scipy import ndimage, misc
#import matplotlib.pyplot as plt

def zoom(img, factor):
    zoom_tuple = (factor,) * 2 + (1,) * (img.ndim - 2)

    return ndimage.zoom(img, zoom_tuple)
