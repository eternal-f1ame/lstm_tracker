import os
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from dataset_utils.kitti_datum import KITTIDataset
from dataset_utils.mot_datum import MOTDataset
from tracker.tracking import Tracker
from trainer.dataset_info import kitti_classes_reverse
from trainer.helpers import lbtr_to_chw, to_one_hot
from vis_utils.vis_utils import draw_bounding_box_on_image
from vis_utils.vis_datum import datum_with_labels

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('frame_dir', action='store' , type=str, help='Directory containing all the frames to use')
args = parser.parse_args()
tracker = Tracker()

path_source = args.frame_dir
path_text, sub_dir = os.path.normpath(path_source).split(os.path.sep)[:-2], os.path.normpath(path_source).split(os.path.sep)[-1]
path_text = os.path.sep.join(path_text)
path_text = os.path.join(path_text, "exp",sub_dir)
print(path_text)
def get_boxes_from_datum(dat, kitti=True):
    i_w, i_h = dat.image.size

    if len(dat.objects) == 0:
        return np.zeros(shape=(0, 4 + 9))

    if kitti:
        _boxes = lbtr_to_chw(
            np.array([[x.x_min / i_w, x.y_min / i_h, x.x_max / i_w, x.y_max / i_h,
            kitti_classes_reverse[x.category]] for x in dat.objects])
            )
    else:
        _boxes = lbtr_to_chw(
            np.array([[x.x_min / i_w, x.y_min / i_h, x.x_max / i_w, x.y_max / i_h, 0]
            for x in dat.objects])
            )

    _boxes = to_one_hot(_boxes, 9)

    return _boxes


ious = []
# pbar = tqdm()

# BASE_PATH = "/home/dark/Documents/GitHub/lstm_tracker"
# dataset = KITTIDataset(root_path="{}/data/KITTI_tracking/data_tracking_image_2/training".format(BASE_PATH))
# # dataset = MOTDataset(root_path="{}/data/MOT16/train".format(BASE_PATH))

# seq_num = 17
# print(seq_num)
# seq = dataset.sequences[seq_num]

init = 1

path = f"{path_text}/labels"
frames = []
paths = []
for ind, file_name in enumerate(sorted(os.listdir(path))):
    with open(f"{path}/{file_name}", "r", encoding="utf-8") as f:
        a = f.readlines()
    for i, e in enumerate(a):
        a[i] = a[i].replace("\n", "")
        a[i] = a[i].split()

    a = np.array(a)
    a = a.astype("float")
    paths.append(
        f"{path_source}/{str(ind).zfill(6)}.png")

    frames.append(a)
Tracked = False
for i, _ in enumerate(frames):

    # if i < 235:
    #     continue
    boxes = frames[i]
    # boxes = get_boxes_from_datum(dat=datum, kitti=True)

    if init:
        tracker.initiate_tracks(boxes)
        init = 0
        print("""
        

        Initiated tracks
        
        
        """)
        continue

    preds = tracker.predict()
    MOTA = np.ones((preds.shape[0],10))*-1
    MOTA[:,0] = i
    MOTA[:,2:6] = preds[:,0:4]
    MOTA = MOTA.round(4)
    if Tracked:
        MOTA[:,1] = np.array(track_ids)
        # print(len(track_ids), len(preds))
    tracks = tracker.update(detections=boxes, predictions=preds)

    track_ids = [track.track_id for track in tracks]
    # print(len(track_ids))
    # print(len(preds))
    Tracked = True

    # im = datum.image.copy()
    im = Image.open(paths[i])
    tracker.draw_tracks(im)
    for MOT in list(MOTA.astype(str)):
        with open(f'../MOTA/{sub_dir}_sequence.txt', 'a',encoding='utf-8') as f:
            f.write(", ".join(list(MOT))+"\n")
    im_orig = Image.open(paths[i])
    for box in boxes:
        b_0 = box[0]-box[2]/2
        b_1 = box[1]-box[3]/2
        b_2 = box[0]+box[2]/2
        b_3 = box[1]+box[3]/2
        draw_bounding_box_on_image(im_orig, b_1, b_0, b_3, b_2, use_normalized_coordinates=True)
    # im_orig = datum_with_labels(datum)
    # ax2.imshow(np.array(im_orig))

    # print(np.array(im_orig).shape)
    cv2.imshow("Display", np.array(im))#cv2.resize(np.array(im),(1280,800)))
    cv2.imshow("Display2", np.array(im_orig))#cv2.resize(np.array(im_orig),(1280,800)))

    cv2.waitKey(1)
    # plt.waitforbuttonpress()
    # plt.pause(1)
    # plt.close()

    # from vis_utils.vis_datum import datum_with_labels
    # datum_with_labels(datum).show()
