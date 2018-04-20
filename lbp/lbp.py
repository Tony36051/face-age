from skimage import feature as ft
from skimage import io
from skimage import transform
import argparse
import utils
import math
import multiprocessing
import os
import tqdm
import numpy as np
import pandas as pd


def extract_lbp(img_full_path):
    image = io.imread(img_full_path, as_grey=True)
    image = transform.resize(image, (128, 128), mode="reflect")
    radius = 2
    n_points = 8 * radius
    features = ft.local_binary_pattern(image, n_points, radius, "uniform")
    return features.astype(np.float32)


def task(data_dir, stage, paths, ages, position):
    file_name = os.path.join(data_dir, "%s_%d.txt" % (stage, position))
    with open(file_name, 'w') as f:
        for i, img_path in enumerate(paths):
            img_path = img_path[1:] if img_path[0] == "/" else img_path
            features = extract_lbp(os.path.join(data_dir, img_path))
            features = np.ravel(features)
            line = ",".join([str(a) for a in features]) + "," + str(ages[i]) + "\n"
            f.write(line)
    return file_name


def merge_pool_results(data_dir, results):
    merged_file = os.path.join(data_dir, "lbp.txt")
    npbin_file = os.path.join(data_dir, "lbp")
    np_list = list()
    with open(merged_file, 'w') as wf:
        for file in results:
            pd_file = pd.read_csv(file, header=None)
            np_list.append(np.array(pd_file))
            with open(file, 'r') as rf:
                wf.writelines(rf.readlines())
    fea_array = np.vstack(tuple(np_list))
    np.save(npbin_file, fea_array)
                # os.remove(file)


def simple_process(data_dir, meta_file):
    """not fast enough"""
    paths = list()
    ages = list()
    with open(os.path.join(data_dir, meta_file), 'r') as f:
        for line in f.readlines():
            ss = line.split(" ")
            paths.append(ss[0])
            ages.append(ss[1])
    with open(os.path.join(data_dir, "lbp_all.txt"), 'w') as f:
        for i in tqdm.trange(len(paths)):
            feature = extract_lbp(os.path.join(data_dir, paths[i]))
            line = ",".join([str(a) for a in feature]) + "," + str(ages[i]) + "\n"
            f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False, help='data dir')
    parser.add_argument('--process', required=False, help='how many process')

    args = parser.parse_args()
    process = int(args.process) if args.process else 16
    data_dir = args.data_dir if args.data_dir else "/home/tony/data"

    stage = "lbp"
    meta_file = "fgnet_label.txt"

    results = list()

    train_path, train_age = utils.read_meta_data(data_dir, meta_file)
    # train_path = train_path[0:16]
    # train_age = train_age[0:16]
    n = int(math.ceil(len(train_path) / float(process)))
    print(len(train_path))

    pool = multiprocessing.Pool(processes=process)
    for i in range(0, len(train_path), n):
        t = pool.apply_async(task, args=(data_dir, stage, train_path[i: i + n], train_age[i:i + n], i,))
        results.append(t)
    pool.close()
    pool.join()

    merge_pool_results(data_dir, [t.get() for t in results])
    [os.remove(os.path.join(data_dir, t.get())) for t in results]
