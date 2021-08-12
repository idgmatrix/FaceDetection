import os
import sys
import glob
import dlib
import cv2
import numpy as np

#predictor_path = 'dlib_dat/shape_predictor_5_face_landmarks.dat'
#predictor_path = 'dlib_dat/shape_predictor_68_face_landmarks.dat'
predictor_path = 'dlib_dat/shape_predictor_68_face_landmarks_GTX.dat'
face_data_path = '../Dataset/CelebA/data512x512\*.jpg'
#face_data_path = '../Dataset/CelebA/celeba\*.jpg'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

cam = cv2.VideoCapture(0)
# OpenCV color format: BGR
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_blue = (255, 0, 0)
line_width = 1

faces = glob.glob(face_data_path)
num_faces = len(faces)
f = open('celeba_front_face_list.txt', 'w')
counter = 0
front_counter = 0
while True:
    # webcam
    ret_val, img = cam.read()
    img = cv2.flip(img, 1)

    #img = dlib.load_rgb_image(faces[counter])
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dets = detector(img)

    for det in dets:
        shape = sp(img, det)

        color = color_green
        rect_color = color_green
        # jaw line
        for i in range(0, 16):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i+1).x, shape.part(i+1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)

        # right eyebrow
        for i in range(17, 21):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i+1).x, shape.part(i+1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        # left eyebrow
        for i in range(22, 26):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i+1).x, shape.part(i+1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)

        # nose
        for i in range(27, 35):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i+1).x, shape.part(i+1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        x0, y0 = shape.part(30).x, shape.part(30).y
        cv2.line(img, (x0, y0), (x2, y2), color, line_width)

        # right eye
        x0, y0 = shape.part(36).x, shape.part(36).y
        for i in range(36, 41):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i + 1).x, shape.part(i + 1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        cv2.line(img, (x0, y0), (x2, y2), color, line_width)
        # left eye
        x0, y0 = shape.part(42).x, shape.part(42).y
        for i in range(42, 47):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i + 1).x, shape.part(i + 1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        cv2.line(img, (x0, y0), (x2, y2), color, line_width)

        # lips
        x0, y0 = shape.part(48).x, shape.part(48).y
        for i in range(48, 59):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i + 1).x, shape.part(i + 1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        cv2.line(img, (x0, y0), (x2, y2), color, line_width)
        # mouth
        x0, y0 = shape.part(60).x, shape.part(60).y
        for i in range(60, 67):
            x1, y1 = shape.part(i).x, shape.part(i).y
            x2, y2 = shape.part(i + 1).x, shape.part(i + 1).y
            cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        cv2.line(img, (x0, y0), (x2, y2), color, line_width)

        cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), rect_color, line_width)
        cx, cy = (det.left() + det.right()) / 2, (det.top() + det.bottom()) / 2
        cv2.circle(img, (int(cx), int(cy)), 2, color)
        #break

    cv2.imshow('Face Detector', img)
    #counter += 1
    #if counter % 100 == 0:
    #    print(counter, front_counter)
    #
    #if num_faces <= counter:
    #    break

    key = cv2.waitKey(1)
    if key == 32 or key == 62 or key == 46:
        counter += 1
    if key == 60 or key == 44:
        counter -= 1
    if key == 27:
        break  # esc to quit

cv2.destroyAllWindows()
f.close()
print('Total counter =', counter)
print('Front Counter =', front_counter)
