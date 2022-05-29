import os
import matplotlib.pyplot as plt
import cv2
path = 'c:/chessResize/br'
destination = 'c:/chessResize/wr'

fig, ax = plt.subplots(1, 2)
for x in os.listdir(path):
    img = cv2.imread(path + '/' + x)
    ax[0].imshow(img)
    value = input('black or white')
    if value == 'w':
        pass
    else:
        pass