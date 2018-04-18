# -*- coding: utf-8 -*-
import cv2
import dlib
import os
import sys
from imutils.face_utils import FaceAligner
from tqdm import tqdm
import math
from multiprocessing import Process as pro, Pool
import argparse
import utils


def opencv_recognize(img_path):
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


def dlib_recognize_and_align(img_path):
    '''加载人脸检测器、加载官方提供的模型构建特征提取器'''
    win = dlib.image_window()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    brg_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(brg_img, cv2.COLOR_BGRA2RGB)
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


def align_and_save(data_dir, img_path):
    full_path = os.path.join(data_dir, img_path)
    bgr_img = cv2.imread(full_path, 0)
    gray = bgr_img
    # gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) != 1:
        return False
    img = fa.align(gray, gray, rects[0])
    new_file_path = os.path.join(data_dir, "aligned", img_path)
    aligned_dir = os.path.dirname(new_file_path)
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)
    cv2.imwrite(new_file_path, img)
    return True


def crop_train_face(train_path, train_age):
    """single process"""
    pre_len = len(train_path)
    new_path = list()
    new_age = list()
    for i in tqdm(range(pre_len)):
        img_path = data_dir + train_path[i]
        new_file_path = align_and_save(img_path)
        new_path.append(new_file_path)
        new_age.append(train_age[i])
    return new_path, new_age


def task(data_dir, db, img_paths, ages, position):
    file_name = os.path.join(data_dir, "%s_%d.txt" % (db, position))
    with open(file_name, 'w') as f:
        for i, img_path in enumerate(img_paths):
            img_path = img_path[1:] if img_path[0] == "/" else img_path
            if align_and_save(data_dir, img_path):
                f.write("aligned/%s %d\n" % (img_path, ages[i]))
    return file_name


def merge_pool_results(data_dir, db, results):
    merged_file = os.path.join(data_dir, "aligned_%s.txt" % db)
    with open(merged_file, 'w') as wf:
        for file in results:
            with open(file, 'r') as rf:
                wf.writelines(rf.readlines())
        os.remove(file)


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False, help='data dir')
    parser.add_argument('--process', required=False, help='how many process')

    args = parser.parse_args()
    process = int(args.process) if args.process else 32
    data_dir = args.data_dir if args.data_dir else "/home/haonan/dqd/data"

    classifier_xml = os.path.join(data_dir, "haarcascade_frontalface_alt2.xml")
    predictor_path = os.path.join(data_dir, "shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    db = "wiki"
    meta_file = "%s.txt" % db
    results = list()
    train_path, train_age = utils.read_meta_data(data_dir, meta_file)
    n = int(math.ceil(len(train_path) / float(process)))
    print(len(train_path))

    pool = Pool(processes=process)
    for i in range(0, len(train_path), n):
        t = pool.apply_async(task, args=(data_dir, db, train_path[i: i + n], train_age[i:i + n], i,))
        results.append(t)
    pool.close()
    pool.join()

    merge_pool_results(data_dir, db, [t.get() for t in results])
    [os.remove(os.path.join(data_dir, t.get())) for t in results]