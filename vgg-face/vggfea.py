from keras.engine import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from tqdm import trange


def read_label(full_path):
    paths = list()
    ages = list()
    with open(full_path, 'r') as f:
        for line in f.readlines():
            ss = line.split(" ")
            paths.append(ss[0])
            ages.append(int(ss[1]))
    return paths, ages


def write_csv(full_path, vgg_feas, ages):
    with open(full_path, 'w') as f:
        for fea, target in zip(vgg_feas, ages):
            line = ",".join(np.char.mod('%f', fea)) + "," + str(target) + "\n"
            f.write(line)


if __name__ == '__main__':

    data_dir = r"/home/tony/data"
    label_path = os.path.join(data_dir, "fgnet_label.txt")
    paths, ages = read_label(label_path)

    model_vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3),
                                 pooling='avg')  # pooling: None, avg or max
    vgg_feas = list()
    for i in trange(len(paths)):
        img_path = os.path.join(data_dir, paths[i])
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        model = model_vgg_features
        features = model.predict(x)
        vgg_feas.append(features)

    fea = np.vstack(tuple(vgg_feas))
    np.save("vgg-fea", fea)
    fea_tar = np.hstack((fea, np.array([ages]).T))
    np.save("vgg-fea-tar", fea_tar)
    # csv_path = os.path.join(data_dir, 'fgnet-vgg-fea.csv')
    # write_csv(csv_path, vgg_feas, ages)
