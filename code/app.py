import cv2 as cv
import os
import argparse
import numpy as np
import time
from flask import Flask, request, Response, render_template, send_file
import base64
from io import BytesIO, StringIO, TextIOWrapper
import json
from urllib.parse import quote
from PIL import Image

# lets set default values for our agrs
parser = argparse.ArgumentParser(
    prog="a function to detect logos using trained yolov4")
parser.add_argument('--weight_file', type=str, default=os.path.abspath(os.path.join(os.getcwd(),
                                                                                    "code/yolo_v4/custom-yolov4-detector_best_1.weights")), help='provide a path to the weight file')
parser.add_argument('--config_file', type=str, default=os.path.abspath(os.path.join(os.getcwd(),
                                                                                    "code/yolo_v4/custom-yolov4-detector.cfg")), help='provide a path to the configuration file')
parser.add_argument('--obj_names', type=str, default=os.path.abspath(os.path.join(
    os.getcwd(), "code/yolo_v4/obj_names")), help='provide a path to the class names file')
# parser.add_argument('--image', type=str, default=os.path.abspath(os.path.join(
#     os.getcwd(), "data/test/adidas_2.jpg")), help='provide a test image file')
parser.add_argument('--confthresh', type=float, default=0.30,
                    help='provide the confidence threshold for detected bounding boxes')
parser.add_argument('--nmsthresh', type=float, default=0.4,
                    help='provide the NMS threshold for overlapping boxes')
args = parser.parse_args()


def img_to_byte_arr(image: Image):
    """convert the test image to bytes array"""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def detector(frame, args=args):
    """perform the detection"""
    logo_detected = False
    net = cv.dnn_DetectionModel(args.config_file, args.weight_file)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    #frame = cv.imread(args.image)

    with open(args.obj_names, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    start = time.time()
    classes, confidences, boxes = net.detect(
        frame, confThreshold=0.3, nmsThreshold=0.4)
    print(f'>>> Yolo took {time.time()-start:.3f} to perform detection')
    # did we detect any logo?
    conf_check = np.asarray(confidences).flatten()
    if conf_check.ndim and conf_check.size:
        logo_detected = True
    for classId, confidence, box in zip(np.asarray(classes).flatten(), np.asarray(confidences).flatten(), boxes):
        label = f'{confidence:.2f}'
        label = f'{names[classId]}: {label}'
        labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        left, top, width, height = box
        top = max(top, labelSize[1])
        cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
        cv.rectangle(frame, (left, top - labelSize[1]), (left +
                                                         labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame, logo_detected


    #cv.imshow('Image', frame)
    # cv.waitKey()
# lets initialize our flask app
UPLOAD_FOLDER = 'static/assets/img'
app = Flask(__name__)
app.secret_key = "tarbo"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    # load the input image
    img = request.files["file"].read()
    img = Image.open(BytesIO(img))
    npimg = np.array(img)
    image = npimg.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    print(f'>>> Image received! Sending to Yolo for detection')
    pred_img, decision = detector(image)
    #file_name = 'predictions.png'
    #cv.imwrite(file_name, pred_img)
    image = cv.cvtColor(pred_img, cv.COLOR_BGR2RGB)
    np_img = Image.fromarray(image)
    img_encoded = img_to_byte_arr(np_img)
    print(f'>>> Sending prediction results back')
    #return Response(response=img_encoded,status=200,mimetype='image/jpeg')
    result = str(base64.b64encode(img_encoded))[2:-1]
    #pred_img=base64(img_encoded).decode("utf-8")
    return render_template("result.html", pred_image=quote(result.rstrip('\n')))
    #return send_file(BytesIO(img_encoded),mimetype='image/jpeg')


@app.route('/')
def home_page():
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
