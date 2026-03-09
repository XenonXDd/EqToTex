from flask import Flask, request, jsonify, redirect, url_for, render_template
from ai import process
import os
import threading


REMOVE_FILE_AFTER = 600 # seconds

app = Flask(__name__)

upload_dir = os.path.join(app.root_path, "upload")
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

for file in os.listdir(upload_dir):
    os.remove(os.path.join(upload_dir, file))
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/api/ask', methods = ['POST'])
def ask():
    image = request.files['image']
    if not image: return "No image provided", 400

    i = 0
    while os.path.exists(os.path.join(app.root_path, "upload", f"image_{i}.png")): i += 1
    
    image_path = os.path.join(app.root_path, "upload", f"image_{i}.png")
    image.save(image_path)

    result = process(image_path)

    threading.Timer(REMOVE_FILE_AFTER, os.remove, args=[image_path]).start() # Remove the image after 10 minutes
    # os.remove(image_path) # Uncomment this line to remove the image immediately after processing
    return jsonify(result = result)



if __name__ == '__main__':
    app.run(debug = True)
