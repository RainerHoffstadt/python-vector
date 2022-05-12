from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import cv2
url = 'http://192.168.178.84/capture'
currentTime = datetime.now()
output = f'data/white/bishop{currentTime.strftime("%Y.%m.%d-%H.%M.%S")}.jpg'
r = requests.get(url)
img = Image.open(BytesIO(r.content))
img.save("dummy.jpg")
img = cv2.imread("dummy.jpg")
x=150
w=100
y=0
h=250
crop_img = img[y:y+h, x:x+w]
#cv2.imshow("image", crop_img)
cv2.imwrite(output, crop_img)
#img = Image.open ("image.jpg")
#img.save(output)