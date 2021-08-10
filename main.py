import sys
import dlib
import cv2

predictor_path = 'dlib_dat/shape_predictor_5_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
color_blue = (255, 0, 0)
line_width = 2
while True:
    ret_val, img = cam.read()
    img = cv2.flip(img, 1)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    #faces = dlib.full_object_detections()
    for det in dets:
        shape = sp(img, det)
        cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        x1, y1 = shape.part(0).x, shape.part(0).y
        x2, y2 = shape.part(1).x, shape.part(1).y
        x3, y3 = shape.part(2).x, shape.part(3).y
        x4, y4 = shape.part(3).x, shape.part(3).y
        x5, y5 = shape.part(4).x, shape.part(4).y
        cv2.line(img, (x1, y1), (x2, y2), color_blue, line_width)
        cv2.line(img, (x2, y2), (x5, y5), color_blue, line_width)
        cv2.line(img, (x5, y5), (x4, y4), color_blue, line_width)
        cv2.line(img, (x4, y4), (x3, y3), color_blue, line_width)
        break


    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
