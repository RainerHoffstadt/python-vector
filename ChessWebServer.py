import os
import requests
import torch
from flask import Flask, jsonify, request
from io import BytesIO
from PIL import Image
from torchvision import transforms

from ChessModel import transfer_model, ChessClasses
import cv2

def load_model():
    m = transfer_model
    m.load_state_dict(torch.load("./tmp/simplenet.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()

img_transfors = transforms.Compose([
    transforms.Resize((100, 250)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def status():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=['GET','POST'])
    def predict():
        if request.method == 'POST':
            img_url = request.form.image_url
        else:
            img_url = request.args.get('image_url', "")


        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img.save("dummy.jpg")
        img = cv2.imread("dummy.jpg")
        x = 150
        w = 100
        y = 0
        h = 250
        crop_img = img[y:y + h, x:x + w]
        # cv2.imshow("image", crop_img)
        cv2.imwrite("image.jpg", crop_img)
        out_img = Image.open("image.jpg")
        img_tensor = img_transfors(out_img).unsqueeze(0)
        prediction = model(img_tensor)

        predicted_class = ChessClasses[torch.argmax(prediction)]
        return jsonify({"image": img_url, "prediction": predicted_class})
        #pre = str(torch.argmax(prediction))
        #return jsonify({"image": img_url, "prediction": pre})


    app.run(host="0.0.0.0")
create_app()