"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
import random
from PIL import Image
import pandas as pd

import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"

# List of models
models = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    
        # # Selected model
    selected_model = random.choice(models)
            # Load model
    try:
        model = torch.hub.load('ultralytics/yolov5', selected_model, pretrained=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return jsonify({"error": "Model loading failed"}), 500

    # Perform inference
    try:
        img = Image.open(io.BytesIO(file.read()))  # Convert bytes to image
        results = model(img, size=640)  # Perform inference
        return jsonify(results.pandas().xyxy[0].to_dict())  # Return inference in JSON format
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({"error": "Inference failed"}), 500

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "Unexpected error occurred"}), 500

      
    # # Selected model
    # selected_model = random.choice(models)
    
    # # load model
    # model = torch.hub.load('ultralytics/yolov5', selected_model)
    # print(file)
    # img = Image.open(io.BytesIO(file.read()))  # Convert bytes to image
    # results = model(img, size=640)  # Perform inference

    # return jsonify(results.pandas().xyxy[0].to_dict())  # Return inference in JSON format

    # if not request.method == "POST":
    #     return
    
    # # Selected model
    # selected_model = random.choice(models)
    # print(f"Selected model: {selected_model}")
    
    # # load model
    # model = torch.hub.load('ultralytics/yolov5', selected_model)
    
    # if request.files.get("image"):
    #     image_file = request.files["image"]
    #     image_bytes = image_file.read()
    #     img = Image.open(io.BytesIO(image_bytes))
    #     results = model(img, size=640) # reduce size=320 for faster inference
    #     return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    # parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()

    # parser.add_argument('--model', default='yolov5l', help='model to run, i.e. --model yolov5s')

    # app.run(host="0.0.0.0", port=args.port)  
    # debug=True causes Restarting with stat
