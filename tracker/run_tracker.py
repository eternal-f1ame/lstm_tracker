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
FEED = 0
path = f"{path_text}/labels"
TRACKED = False
frames = []
paths = []
i = 0
while True:
    print(i, "WHILE")
    file_name = str(i).zfill(6)+".txt"
    with open(f"{path}/{file_name}", "r", encoding="utf-8") as f:
        a = f.readlines()
    for ind, e in enumerate(a):
        a[ind] = a[ind].replace("\n", "")
        a[ind] = a[ind].split()


    a = np.array(a)
    a = a.astype("float")
    boxes = a

    if INIT:
        tracker.initiate_tracks(boxes)
        INIT = 0
        print("""
        

        Initiated tracks
        
        
        """)
        print(i, "INIT")

        continue

    path_im = f"{path_source}/{str(i).zfill(6)}.png"
    preds = tracker.predict()

    tracks = tracker.update(detections=boxes, predictions=preds)

    with open("../COMM", "w", encoding="utf-8") as f:
        f.write(str(i+1))

    with open("../FEED",'r',encoding="utf-8") as f:
        try:
            FEED = int(f.read())
        except:
            pass
    
    while FEED-1<i and FEED != -1:
        try:
            with open("../FEED",'r',encoding="utf-8") as f:
                FEED = int(f.read())
            continue
        except Exception as _e:
            continue



    im = Image.open(path_im)
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

    im_orig = Image.open(path_im)
    for box in boxes:
        b_0 = box[0]-box[2]/2
        b_1 = box[1]-box[3]/2
        b_2 = box[0]+box[2]/2
        b_3 = box[1]+box[3]/2
        draw_bounding_box_on_image(im_orig, b_1, b_0, b_3, b_2, use_normalized_coordinates=True)

    if args.view_tracking:
        cv2.imshow("Tracking", np.array(im))#cv2.resize(np.array(im),(1280,800)))

    if args.view_gt:
        print(path_im)
        cv2.imshow("Ground truth", np.array(im_orig))#cv2.resize(np.array(im_orig),(1280,800)))

    cv2.waitKey(1)
    i += 1
    if FEED == -1:
        with open("../COMM", "w", encoding="utf-8") as f:
            f.write(str(0))

        break
