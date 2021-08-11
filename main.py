import os
import sys
import glob
import dlib
import cv2

predictor_path = 'dlib_dat/shape_predictor_5_face_landmarks.dat'
face_data_path = 'I:\Dataset\CelebA\data512x512\*.jpg'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

cam = cv2.VideoCapture(0)
# OpenCV color format: BGR
color_red = (0, 0, 255)
color_green = (0, 255, 0)
color_blue = (255, 0, 0)
line_width = 2

faces = glob.glob(face_data_path)
num_faces = len(faces)
f = open('front_face_list.txt', 'w')
counter = 0
front_counter = 0
while True:
    #ret_val, img = cam.read()
    img = dlib.load_rgb_image(faces[counter])
    #img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dets = detector(img)
    #dets = detector(img)

    for det in dets:
        shape = sp(img, det)
        x1, y1 = shape.part(0).x, shape.part(0).y
        x2, y2 = shape.part(1).x, shape.part(1).y
        x3, y3 = shape.part(2).x, shape.part(3).y
        x4, y4 = shape.part(3).x, shape.part(3).y
        x5, y5 = shape.part(4).x, shape.part(4).y

        color = color_blue
        rect_color = color_green
        if abs(y2 - y4) > 5:
            color = color_red
            rect_color = color_red

        if abs(x5 - (x2 + x4) / 2) > 6:
            color = color_red
            rect_color = color_red

        eye_width = abs(x1 - x3)
        eye_height = abs((y1 + y3) / 2 - y5)
        ratio = eye_height / eye_width
        text = '{}'.format(ratio)
        if ratio < 0.45:
            color = color_red
            rect_color = color_red

        if color == color_blue:
            file_name = faces[counter]
            base_name = os.path.basename(file_name) + '\n'
            f.write(base_name)
            front_counter += 1

        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 1, cv2.LINE_AA)

        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
        cv2.line(img, (x2, y2), (x5, y5), color, line_width)
        cv2.line(img, (x5, y5), (x4, y4), color, line_width)
        cv2.line(img, (x4, y4), (x3, y3), color, line_width)
        cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), rect_color, line_width)
        cx, cy = (det.left() + det.right()) / 2, (det.top() + det.bottom()) / 2
        cv2.circle(img, (int(cx), int(cy)), 2, color)
        #break

    cv2.imshow('Face Detector', img)
    counter += 1
    if counter % 100 == 0:
        print(counter, front_counter)

    if num_faces <= counter:
        break

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
