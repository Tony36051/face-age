# -*- coding: utf-8 -*-
import logging
from utils import get_meta
from tqdm import trange
import numpy as np
import os
import sys

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


def main_process(data_path, db):
    mat_path = os.path.join(data_path, db + "_crop", db + ".mat")
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    ok_idx = filter_unusual(full_path, gender, face_score, second_face_score, age)
    with open(db + ".txt", 'w') as f:
        for i in trange(len(ok_idx)):
            f.write("%s/%s %d\n" % (db, full_path[i][0], age[i]))




if __name__ == '__main__':
    data_path = str(sys.argv[1]) if len(sys.argv) > 1 else r"d:/data"
    db = "wiki"
    main_process(data_path, db)
