# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy
import os
from imutils.face_utils import FaceAligner
from tqdm import tqdm, trange
import multiprocessing
from time import sleep, time
import threading
import math
#cur_dir = os.path.dirname(__file__)
cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data")
#data_dir = r"d:/data"  # special for windows
classifier_xml = os.path.join(data_dir, "haarcascade_frontalface_alt2.xml")
predictor_path = os.path.join(data_dir, "shape_predictor_68_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=160)

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
    rgb_img = cv2.cvtColor(brg_img,  cv2.COLOR_BGRA2RGB)
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


def align_and_save(img_path):

    bgr_img = cv2.imread(img_path)
    # gray = bgr_img
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) != 1:
        return None
    img = fa.align(gray, gray, rects[0])
    new_file_path = img_path.replace("data", "aligned")
    dir = os.path.dirname(new_file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    cv2.imwrite(img_path.replace("data", "aligned"), img)
    return new_file_path

def read_pre_age_txt(file_name):
    path = list()
    age = list()
    with open(os.path.join(data_dir, file_name)) as f:
        for line in f.readlines():
            ss = line.split(" ")
            path.append(ss[0])
            age.append(int(ss[1]))
    return path, age


def crop_train_face(train_path, train_age):
    pre_len = len(train_path)
    new_path = list()
    new_age = list()
    for i in tqdm(range(pre_len)):
        img_path = data_dir + train_path[i]
        new_file_path = align_and_save(img_path)
        new_path.append(new_file_path)
        new_age.append(train_age[i])
    return new_path, new_age


def task(position, train_path, train_age):
    # while True:
    #     a = 100
    #     a = a ** a
    # a  = len(start)
    # for img_path in train_path:
    #     # index = i + start
    #     r = align_and_save(img_path = data_dir + img_path)
    #     if r is None:
    #         train_age[i] = -1
    with open("pos_"+str(position)+".txt", 'w') as f:
        for i, img_path in enumerate(train_path):
            # index = i + start
            #print(data_dir, img_path)
            img_path = data_dir + img_path
            
            #img_path = os.path.join(data_dir, img_path)
            #img_path = os.path.normpath(img_path)

            #if not os.path.isfile(img_path):
            #    print("file not existed:"+img_path)
            r = align_and_save(img_path)
            if r is None:
                train_age[i] = -1
            else:
                f.write("%s %d"%(img_path, train_age[i]))

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

from multiprocessing import Process as pro
from multiprocessing.dummy import Process as thr
def run(i):
	lists=range(i)
	list(set(lists))
def multicore():
    '''
    	多进程
    	'''
    for i in range(0, len(train_path), n):  ##10-2.1s 20-3.8s 30-5.9s
        t = pro(target=task, args=(i,i+n,))
        t.start()

def ok():
    a = r"d:/data/imdb_crop/49/nm0082749_rm3881159680_1971-1-18_2008.jpg"
    img = cv2.imread(a)
    cv2.imshow("img", img)
    exit(0)

if __name__ == '__main__':
    # ok()

    train_path, train_age = read_pre_age_txt("train_age.txt")
    # print(len(train_path))
    # new_path, new_age = crop_train_face(train_path, train_age)
    # print(len(train_path), len(new_age))

    m = 16
    n = int(math.ceil(len(train_path) / float(m)))
    # pool = multiprocessing.Pool(processes=m)
    # for i in range(0, len(train_path), n):  ##10-2.1s 20-3.8s 30-5.9s
    #     pool.apply(task, args=(train_age[i: i+n]))
    # pool.close()
    # pool.join()
    # retlist = [pool.apply_async(task(i,i + n)) for i in range(0, len(train_path), n)]
    #
    # print('Waiting for all subprocesses done...')
    # pool.close()
    # pool.join()
    # print('All subprocesses done.')

    for i in range(0, len(train_path), n):  ##10-2.1s 20-3.8s 30-5.9s
        t = pro(target=task, args=(i, train_path[i: i+n],train_age[i:i+n]))
        t.start()









