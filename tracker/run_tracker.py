import os
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tracker.tracking import Tracker, _NO_MATCH
from vis_utils.vis_utils import draw_bounding_box_on_image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    'frame_dir', action='store' , type=str,
    help='Directory containing all the frames to use'
    )
parser.add_argument('--view_gt', action='store_true')
parser.add_argument('--view_tracking', action='store_true')
parser.add_argument('--save_tracking', action='store_true')
args = parser.parse_args()
tracker = Tracker()

path_source = args.frame_dir
PATH_TEXT = os.path.sep.join(os.path.normpath(path_source).split(os.path.sep)[:-2])
sub_dir = os.path.normpath(path_source).split(os.path.sep)[-1]
path_text = os.path.join(PATH_TEXT, "exp",sub_dir)

ious = []
INIT = 1

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

TRACKED = False

for i, _ in enumerate(frames):
    boxes = frames[i]

    if INIT:
        tracker.initiate_tracks(boxes)
        INIT = 0
        print("""
        

        Initiated tracks
        
        
        """)
        continue

    preds = tracker.predict()

    tracks = tracker.update(detections=boxes, predictions=preds)

    im = Image.open(paths[i])
    tracker.draw_tracks(im)

    if args.save_tracking:
        for track in tracks:

            if track.state == _NO_MATCH:
                continue

            tracks_pos = track.to_tlbr()[-1, :4]
            out_str = f"{i} {track.track_id}"+" {} {} {} {} \n".format(*tracks_pos)
            with open(
                f'../data/results/{sub_dir}.txt',
                'a+',encoding='utf-8') as f:
                f.write(out_str)

    im_orig = Image.open(paths[i])
    for box in boxes:
        b_0 = box[0]-box[2]/2
        b_1 = box[1]-box[3]/2
        b_2 = box[0]+box[2]/2
        b_3 = box[1]+box[3]/2
        draw_bounding_box_on_image(im_orig, b_1, b_0, b_3, b_2, use_normalized_coordinates=True)

    if args.view_tracking:
        cv2.imshow("Tracking", np.array(im))#cv2.resize(np.array(im),(1280,800)))

    if args.view_gt:
        cv2.imshow("Ground truth", np.array(im_orig))#cv2.resize(np.array(im_orig),(1280,800)))

    cv2.waitKey(1)
