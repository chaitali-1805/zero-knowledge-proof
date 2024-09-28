from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Route for the user to upload image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!", 400
        file = request.files['file']
        
        # Send the image to the yolov5 API
        files = {'file': file.read()}
        
        response = requests.post('http://hiwi-yolov5-1:5000//v1/object-detection/yolov5', files=files)
        
        if response.status_code == 200:
            # Pass the response to a template to display it to the user
            result = response.json()
            return render_template('result.html', result=result)
        elif response.status_code == 500:
            return "Server error"
        else:
            return "Failed to get prediction from yolov5"

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
