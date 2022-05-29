from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import cv2
from zoom import zoom
url = 'http://192.168.178.83/capture'
currentTime = datetime.now()
output = f'data/white/bishop{currentTime.strftime("%Y.%m.%d-%H.%M.%S")}.jpg'
r = requests.get(url)
img = Image.open(BytesIO(r.content))
#img.save(output)

img.save("dummy.jpg")
img = cv2.imread("dummy.jpg")

x=570
w=200
y=300
h=200
crop_img = img[y:y+h, x:x+w]
crop_img = zoom(crop_img, 10)
#cv2.imshow("image", crop_img)
cv2.imwrite(output, crop_img)
#img = Image.open ("image.jpg")
#img.save(output)