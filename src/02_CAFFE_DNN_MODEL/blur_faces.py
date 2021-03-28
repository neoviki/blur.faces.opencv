'''
    Blur Faces based on DNN - Caffe Model

    Author  : Viki (a) Vignesh Natarajan
    Contact : vikiworks.io
'''


import cv2
import time
from imutils.video import VideoStream
import numpy as np
import imutils

neural_network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

def get_video_from_webcam():
    return VideoStream(src=0).start()

def get_image_frame(video_stream):
    return video_stream.read()

def resize_image(image, width):
    return imutils.resize(image, width=width)


def image2blob(image):
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

def detect_faces(image_blob):
    neural_network.setInput(image_blob)
    return neural_network.forward()

def add_label(image, label, x, y):
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

def blur_faces(image_frame, faces):
    (h, w) = image_frame.shape[:2]
    for i in range(0, faces.shape[2]):
        accuracy = faces[0, 0, i, 2]

        if accuracy < 0.5:
            continue

        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        y = startY - 10
        if y < 10:
            y = startY + 10
      
        #Draw Bounding Box
        #cv2.rectangle(image_frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

        face = image_frame[startY:endY,startX:endX]
        blurred_face = blur_image(face)
        image_frame[startY:endY,startX:endX] = blurred_face

        label = "{:.2f}%".format(accuracy * 100)
        add_label(image_frame, label, startX, y)

    return image_frame

def blur_image(image):
        factor=1.0
        (h, w) = image.shape[:2]
        k_w = int(w / factor)
        k_h = int(h / factor)

        if k_w % 2 == 0:
            k_w = k_w - 1

        if k_h % 2 == 0:
            k_h = k_h - 1
	
        return cv2.GaussianBlur(image, (k_w, k_h), 0)

def is_exit_key_pressed():
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        return True
    else:
        return False

def display_image(image):
    cv2.imshow('Frame', image)

def main():
    video_stream = get_video_from_webcam()
    time.sleep(1.0)

    while True:
        image_frame = get_image_frame(video_stream)
        image_frame = resize_image(image_frame, 400)

        blob = image2blob(image_frame)

        faces = detect_faces(blob)

        image_frame = blur_faces(image_frame, faces)

        display_image(image_frame)

        if is_exit_key_pressed() == True:
            cv2.destroyAllWindows()
            video_stream.stop()

main()
