# utils.py
# -*- coding: utf-8 -*-
from scipy.io import loadmat
from datetime import datetime
import os
import numpy as np


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    # age = np.array(map(calc_age, photo_taken, dob))  # python 2/3 staff
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass


def read_meta_data(data_dir, file_name):
    path = list()
    age = list()
    with open(os.path.join(data_dir, file_name)) as f:
        for line in f.readlines():
            ss = line.split(" ")
            path.append(ss[0])
            age.append(int(ss[1]))
    return path, age


if __name__ == '__main__':
    mat_path = r"D:\wiki_crop\wiki.mat"
    db = "wiki"
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
