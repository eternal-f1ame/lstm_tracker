import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from dataset_utils.kitti_datum import KITTIDataset
from dataset_utils.mot_datum import MOTDataset
from tracker.tracking import Tracker
from trainer.dataset_info import kitti_classes_reverse
from trainer.helpers import lbtr_to_chw, to_one_hot

from vis_utils.vis_datum import datum_with_labels

tracker = Tracker()


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
pbar = tqdm()

BASE_PATH = "/home/dark/Documents/GitHub/lstm_tracker"
dataset = KITTIDataset(root_path="{}/data/KITTI_tracking/data_tracking_image_2/training".format(BASE_PATH))
# dataset = MOTDataset(root_path="{}/data/MOT16/train".format(BASE_PATH))

seq_num = 6
print(seq_num)
seq = dataset.sequences[seq_num]
out = cv2.VideoWriter(f'video/output_video{seq_num}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, (1200,800))

init = 1

for i, datum in enumerate(seq.datums()):

    # if i < 235:
    #     continue

    boxes = get_boxes_from_datum(dat=datum, kitti=True)

    if init:
        tracker.initiate_tracks(boxes)
        init = 0
        continue

    preds = tracker.predict()
    tracker.update(detections=boxes, predictions=preds)
    im = datum.image.copy()
    tracker.draw_tracks(im)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # ax1.imshow(np.array(im))
    ax1.axis("off")

    im_orig = datum_with_labels(datum)
    # ax2.imshow(np.array(im_orig))
    ax2.axis("off")
    # print(np.array(im_orig).shape)
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Display", cv2.resize(np.array(im),(1280,800)))
    cv2.imshow("Display2", cv2.resize(np.array(im_orig),(1280,800)))
    out.write(cv2.resize(np.array(im),(1200,800)))

    cv2.waitKey(1)
    # plt.waitforbuttonpress()
    # plt.pause(1)
    # plt.close()

    # from vis_utils.vis_datum import datum_with_labels
    # datum_with_labels(datum).show()
