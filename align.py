# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy
import os

cur_dir = os.path.dirname(__file__)
data_dir = os.path.join(cur_dir, "data")
classifier_xml = os.path.join(data_dir, "haarcascade_frontalface_alt2.xml")
predictor_path = os.path.join(data_dir, "shape_predictor_68_face_landmarks.dat.bz2")
img_path = os.path.join(data_dir, "wiki_crop", "00", "23300_1962-06-19_2011.jpg")


def demo_one(img_path):
    face_patterns = cv2.CascadeClassifier(classifier_xml)
    sample_image = cv2.imread(img_path)
    # cv2.imshow("image", sample_image)
    faces = face_patterns.detectMultiScale(sample_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imwrite('/Users/abel/201612_detected.png', sample_image)
    cv2.imshow("image", sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def dlib_one(img_path):
    '''加载人脸检测器、加载官方提供的模型构建特征提取器'''
    win = dlib.image_window()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    brg_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(brg_img, cv2.COLOR_BRG2RGB)
    dets = detector(rgb_img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(rgb_img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
    pass

if __name__ == '__main__':
    # img_path = r"D:\wiki_crop\00\23300_1962-06-19_2011.jpg"
    # demo_one(img_path)
    dlib_one(img_path)