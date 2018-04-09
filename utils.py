# utils.py
# -*- coding: utf-8 -*-
from scipy.io import loadmat
from datetime import datetime
import os
import plot
import logging
from tqdm import tqdm
import numpy as np

MAX_AGE = 117
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass


def filter_unusual(full_path, gender, face_score, second_face_score, age):
    unusual_gender_idx = list()
    unusual_face_score_idx = list()
    unusual_second_face_score_idx = list()
    unusual_age_idx = list()
    length = len(full_path)
    for i in tqdm(range(length)):
        if np.isnan(gender[i]) or ~(0 <= gender[i] <= 1):
            unusual_gender_idx.append(i)
            logger.warn("unusual gender: %d, %s" % (i, str(gender[i])))

        if face_score[i] < 0:
            unusual_face_score_idx.append(i)
            logger.warn("no face: %d, %.2f" % (i, face_score[i]))

        if second_face_score[i] > 0:
            unusual_second_face_score_idx.append(i)
            logger.warn("more than one face: %d, %.2f" % (i, second_face_score[i]))

        if ~(0 <= age[i] <= MAX_AGE):
            unusual_age_idx.append(i)
            logger.warn("unusual age: %d, %d" % (i, age[i]))
    print(1)


if __name__ == '__main__':
    mat_path = ur"D:\wiki_crop\wiki.mat"
db = "wiki"
full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

filter_unusual(full_path, gender, face_score, second_face_score, age)

age = [a for a in age if 0 < a < MAX_AGE]
print(len(age))
plot.histgram_demo(age)