import os
from collections import defaultdict
import random
import numpy as np
import tensorflow as tf
tf = tf.compat.v1
from matplotlib import pyplot as plt, patches
from vis_utils.vis_datum import ImageBoxes



def MOTA_to_train(path="../data/MOTA/",
                 split="train", testing=False, one_hot_classes=False, anchors=False, num_classes=9):
    path_l = os.listdir(path)
    random.shuffle(path_l)
    for file_name in path_l:
        text_file = open(path+file_name,"r",encoding="utf-8").read().splitlines()
        tracks = defaultdict(dict)
        text_file_arr = np.array([line.rstrip().split() for line in text_file]).astype(float)
        for i in range(int(max(text_file_arr[:,0]))):
            tracked = defaultdict(dict)
            track = text_file_arr[text_file_arr[:,1]==i][:,2:7]

            for j in range(len(track)):
                tracked[j]['x'] = track[j,0]
                tracked[j]['y'] = track[j,1]
                tracked[j]['h'] = track[j,2]
                tracked[j]['w'] = track[j,3]
                tracked[j]['class'] = track[j,4]
            tracks[i] = tracked

        valid_tracks = list(tracks.keys())
        if split == "train":
            np.random.shuffle(valid_tracks)

        for track_id in valid_tracks:
            track = tracks[track_id]
            f_step = len(track.keys())
            if f_step < 11:
                continue
            x = np.zeros(shape=(10, 5), dtype=np.float32)
            if testing:
                x_im = []
            for start in range(0, f_step - 11):
                l_step = int(sorted(list(track.keys()), key=lambda a: int(a))[start]) - 1
                i = 0
                for t_step, data in sorted(list(track.items())[start:start + 10], key=lambda vid: int(vid[0])):
                    assert int(t_step) > l_step, "order error; keys t-{} & l-{}".format(int(t_step), l_step)
                    l_step = int(t_step)

                    x[i] = np.array([data['x'], data['y'], data['h'], data['w'], data['class']])
                    if testing:
                        x_im.append(data['im'])

                    i += 1
                _, data = list(track.items())[start + 10]
                y = np.array([data['x'], data['y'], data['h'], data['w'], data['class']], dtype=np.float32)
                if testing:
                    y_im = data["im"]
                    if one_hot_classes:
                        temp = np.zeros(shape=(x.shape[0], num_classes))
                        temp[np.array(range(len(x[:, 4]))), x[:, 4].astype(int)] = 1
                        x_ = np.concatenate([x[:, :4], temp.astype(float)], axis=-1)
                        assert x_.shape == (10, 4 + num_classes), "wrong shape for x"

                    if anchors:
                        y_x, y_y, y_h, y_w = (y[:4] - x[-1, :4])
                        y_x, y_y = make_anchors(y_x, (-0.5, 0, 0.1, 0.2, 0.5)), make_anchors(y_y, (-0.5, 0, 0.1, 0.2, 0.5))
                        y_h, y_w = make_anchors(y_h, (-0.5, 0, 0.1, 0.2, 0.5)), make_anchors(y_w, (-0.5, 0, 0.1, 0.2, 0.5))
                        y = np.array([y_x, y_y, y_h, y_w])

                if testing:
                    if one_hot_classes:
                        yield (x_, y, x_im, y_im)
                    else:
                        yield (x, y, x_im, y_im)
                else:
                    assert x.shape == (10, 5), "invalid shape"
                    yield(x, y)

    

def make_anchors(val, anchor_centres=(-0.5, 0, 0.1, 0.2, 0.5)):
    """

    Args:
        val:                value to anchor
        anchor_centres:     tuple of anchor centres

    Returns:
        np.array of shape (6,) containing confidence and distance to anchor centre respectively.
        output[:3] is confidence, and output[3:] is distance.
    """
    idx = np.argmin(np.abs(val - np.array(anchor_centres)))
    conf = np.zeros(shape=len(anchor_centres))
    dist = np.zeros(shape=len(anchor_centres))
    conf[idx] = 1
    dist[idx] = val - anchor_centres[idx]

    return np.concatenate((conf, dist), axis=0)


def joint_data_gen(paths=("../data/MOTA/","../data/MOTA/"),
                   split="train", num_classes=9, anchors=True, one_hot_classes=True):
    """

    Args:
        paths:              list/tuple of str to KITTI path, MOT path respectively
        split:              train / val
        num_classes:        number of classes
        anchors:            output y as anchors
        one_hot_classes:    output x with classes in one hot encoding

    Returns:
        generator with each iteration yielding a tuple containing (x,y), ie the ground truth and label for a track.
        If anchors, output y is of shape (4, 6). 6 corresponds to 3 anchors confidence and distance respectively. Else,
        shape is (4,) for [centre_x, centre_y, height, width].
        If one_hot_classes, output x is of shape (10, 4 + num_classes). Else, shape is (10, 5). 10 corresponds to the
        time steps in both cases.
    """
    if type(split) == bytes:
        split = split.decode("utf-8")
    assert split in ["train", "val"], "invalid split type: {}".format(split)

    gens = (MOTA_to_train, MOTA_to_train)
    gens = [gen(path=path, split=split) for gen, path in zip(gens, paths)]
    while True:

        try:
            x, y = next(gens[1])
        except StopIteration:
            gens[1] = MOTA_to_train(path=paths[1], split=split)
            x, y = next(gens[1])

        if one_hot_classes:
            temp = np.zeros(shape=(x.shape[0], num_classes))
            temp[np.array(range(len(x[:, 4]))), x[:, 4].astype(int)] = 1
            x = np.concatenate([x[:, :4], temp.astype(float)], axis=-1)
            assert x.shape == (10, 4 + num_classes), "wrong shape for x"

        if anchors:
            y_x, y_y, y_h, y_w = (y[:4] - x[-1, :4])
            y_x, y_y = make_anchors(y_x, (-0.5, 0, 0.1, 0.2, 0.5)), make_anchors(y_y, (-0.5, 0, 0.1, 0.2, 0.5))
            y_h, y_w = make_anchors(y_h, (-0.5, 0, 0.1, 0.2, 0.5)), make_anchors(y_w, (-0.5, 0, 0.1, 0.2, 0.5))
            y = np.array([y_x, y_y, y_h, y_w])
        else:
            y = y[:4]
        yield x, y





def val_data_gen(paths=("../data/MOTA",
                        "../data/MOTA"),
                 split="train", num_classes=9):
    """
    Method to return images for validation data gen.
    """
    gens = (MOTA_to_train, MOTA_to_train)
    gens = [gen(path=path, split=split, testing=True, one_hot_classes=True, anchors=True, num_classes=num_classes)
            for gen, path in zip(gens, paths)]
    while True:
        a = np.random.choice(range(4))  # MOT has over 3 times tracks as KITTI
        if a < 1:
            x, y, x_im, y_im = next(gens[1])
        else:
            x, y, x_im, y_im = next(gens[0])

        yield x, y


def to_bbox(x, y):
    idx = np.argmax(y[:, :3], axis=-1)
    dist = y[:, 3:][np.array(range(len(idx))), idx]
    y = (dist + idx) * (x[-1, :4] - x[-2, :4] + 1e-5) + x[-1, :4]
    return y


def vis_gen(auto_time=False):
    gen = MOTA_to_train(testing=True)
    while True:
        x, y, x_im, y_im = next(gen)
        fig, ax = plt.subplots(1)
        image = ImageBoxes(path=y_im)
        plt.axis("off")
        for num, i in enumerate(x):
            im = plt.imread(x_im[num])
            ih, iw, _ = im.shape
            cx, cy, h, w = i[:4]
            image.add_box([cx, cy, w, h], color='blue')
        for i in [y]:
            im = plt.imread(y_im)
            ih, iw, _ = im.shape
            cx, cy, h, w = i[:4]
            image.add_box([cx, cy, w, h])
        ax.imshow(np.array(image.get_final()))
        if auto_time:
            plt.pause(0.1)
        else:
            plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":
    vis_gen()
