import numpy as np
import cv2
from flask import Flask, request
from flask import render_template
from detect import predictPlayer
from detectCam import detect_WebCam_video


#INITALIZING THE FLASK APPLICATION
app = Flask(__name__)

#SETTING UP DEFAULT ROUTE TO INDEX PAGE
@app.route("/")
def hello_world():
    return render_template('index.html')


#ROUTE TO THE DOWNLAOD
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    return render_template('download.html')


#ROUTE TO CAM DETECTION
@app.route('/mainCam', methods=['POST', 'GET'])
def mainCam():
    if request.method == 'POST':
        print("jdjskjvkjdkjvksjkdsjkdvjkjdk")
        #PASSES THE CALL TO WEBCAM DETECTION METHOD IN DETECTCAM.PY 
        detect_WebCam_video()
    return render_template('cam.html')


#ROUTE TO THE MAIN PAGE OF APPLICATION
@app.route('/main', methods=['POST', 'GET'])
def main():
    #PASSES THE IMAGE FROM POST METHOD TO PLAYER PREDICTION ETHOD IN DETECT.PY 
    if request.method == 'POST':
        img = request.files['image'].read()
        file_bytes = np.fromstring(img, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        print("jdjskjvkjdkjvksjkdsjkdvjkjdk")
        print("Path is ",img)

        result = predictPlayer(img)
        print(result, type(result))
        cv2.imshow("ResultImage", result)

    return render_template('download.html')


#STARTING FLASK APPLICATION ON LOCAL HOST
if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)
    #app.run(debug=True, host='0.0.0.0')
    #app.run(debug=True ,port=8080, host='localhost', use_reloader=False)