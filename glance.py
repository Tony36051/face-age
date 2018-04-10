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
    age_idx = [i for i in xrange(len(age)) if 0 <= age[i] <= MAX_AGE]
    # face filter
    # face_score_idx = np.where(face_score <= 0)[0]
    # second_face_score_idx = np.where(second_face_score>0)[0]

    return np.intersect1d(gender_idx, age_idx)


def plot_age_dist():
    mat_path = ur"D:\wiki_crop\wiki.mat"
    db = "wiki"
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    ok_idx = filter_unusual(full_path, gender, face_score, second_face_score, age)

    plot.histgram_demo(age[ok_idx])

if __name__ == '__main__':
    pass
