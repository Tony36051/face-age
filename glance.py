# -*- coding: utf-8 -*-
import logging
from utils import get_meta
from tqdm import tqdm
import numpy as np
import plot

MAX_AGE = 117
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def filter_unusual(full_path, gender, face_score, second_face_score, age):
    # label filter
    gender_idx = np.where(~np.isnan(gender))[0]
    age_idx = [i for i in range(len(age)) if 0 <= age[i] <= MAX_AGE]
    # face filter
    # face_score_idx = np.where(face_score <= 0)[0]
    # second_face_score_idx = np.where(second_face_score>0)[0]

    return np.intersect1d(gender_idx, age_idx)


def plot_age_dist():
    mat_path = r"D:\wiki_crop\wiki.mat"
    db = "wiki"
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    ok_idx = filter_unusual(full_path, gender, face_score, second_face_score, age)

    plot.histgram_demo(age[ok_idx])

if __name__ == '__main__':
    # plot_age_dist()

    img_path = r"FG-NET\027A20.JPG"
    data_dir = r"d:/data"

    import os
    import dlib
    import cv2
    import dlib
    import os
    import matplotlib.pyplot as plt

    classifier_xml = os.path.join(data_dir, "haarcascade_frontalface_alt2.xml")
    predictor_path = os.path.join(data_dir, "shape_predictor_68_face_landmarks.dat")

    import cv2
    import dlib
    from skimage import io

    # io.use_plugin('matplotlib', 'imread')

    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(predictor_path)
    img = io.imread(os.path.join(data_dir, img_path))
    gray = io.imread(os.path.join(data_dir, img_path), as_grey=True)
    # original_image = img.copy()
    faces = detector(img, 1)
    if (len(faces) > 0):
        for k, d in enumerate(faces):
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 3)
            shape = landmark_predictor(img, d)
            for i in range(68):
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1, 1)
                # cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 2555, 255))
    # marked_image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Frame', img)
    # cv2.waitKey(0)

    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(img)
    plt.subplot(133)
    plt.imshow(gray)
    # plt.subplot_tool()
    plt.show()
