from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
url = 'http://192.168.178.84/capture'
currentTime = datetime.now()
output = f'data/white/bishop{currentTime.strftime("%Y.%m.%d-%H.%M.%S")}.jpg'
r = requests.get(url)
img = Image.open(BytesIO(r.content))
img.save(output)