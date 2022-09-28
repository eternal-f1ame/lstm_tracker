from mimetypes import init
import os
import shutil
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict
from tracker.tracking import Tracker, _NO_MATCH
from vis_utils.vis_utils import draw_bounding_box_on_image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    'frame_dir', action='store' , type=str,
    help='Directory containing all the frames to use'
    )
parser.add_argument(
    'yolo', action='store' , type=int,
    help='Directory containing all the frames to use'
    )
parser.add_argument('--view_gt', action='store_true')
parser.add_argument('--view_tracking', action='store_true')
parser.add_argument('--save_tracking', action='store_true')
args = parser.parse_args()
tracker = Tracker()
path_source = args.frame_dir


init_im_path = f"{path_source}/{str(0).zfill(6)}.png"
init_im = Image.open(init_im_path)
init_w, init_h = init_im.size


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

objects = True


real_time = args.yolo


if real_time > -1:
    with open(f"../data/MOTA/{str(real_time).zfill(4)}.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        line_arr = np.array([line.rstrip().split() for line in lines]).astype(float)
        boxes_arr = defaultdict(dict)
        for i in range(int(max(line_arr[:,0]))):
            boxes_arr[i] = line_arr[line_arr[:,0]==i][:,2:]
    f.close()


i = 0
out = cv2.VideoWriter(f'../tracker_testing/results/{sub_dir}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (init_w,init_h))
try:
    while True:

        path_im = f"{path_source}/{str(i).zfill(6)}.png"
        if real_time == -1:
            file_name = str(i).zfill(6)+".txt"
            try:
                with open(f"{path}/{file_name}", "r", encoding="utf-8") as f:
                    a = f.readlines()

                for ind, e in enumerate(a):
                    a[ind] = a[ind].replace("\n", "")
                    a[ind] = a[ind].split()

                a = np.array(a)
                a = a.astype("float")
                boxes = np.c_[a[:,0:-1],tf.one_hot(a[:,-1],depth=9).numpy()]
                objects = True
            except:
                objects = False
                pass


            with open("../COMM", "w", encoding="utf-8") as f:
                f.write(str(i+1))

            with open("../FEED",'r',encoding="utf-8") as f:
                try:
                    FEED = int(f.read())
                except Exception as _e:
                    pass

            while FEED-1<i and FEED != -1:
                try:
                    with open("../FEED",'r',encoding="utf-8") as f:
                        FEED = int(f.read())
                    continue
                except Exception as _e:
                    continue

        else:
            try:

                boxes = np.c_[boxes_arr[i][:,0:-1],tf.one_hot(boxes_arr[i][:,-1],depth=9).numpy()]
                objects = True
            except:
                objects = False
                pass

        if INIT:
            tracker.initiate_tracks(boxes)
            INIT = 0
            print("""
            

            Initiated tracks
            
            
            """)
            continue


        im = Image.open(path_im)
        im_orig = Image.open(path_im)
    

        if objects:
            preds = tracker.predict()
            tracks = tracker.update(detections=boxes, predictions=preds)

            tracker.draw_tracks(im)

            if args.save_tracking:
                for track in tracks:

                    if track.state == _NO_MATCH:
                        continue

                    tracks_pos = track.to_cwh()[-1, :4]
                    out_str = f"{i} {track.track_id}"+" {} {} {} {} \n".format(*tracks_pos)
                    with open(
                        f'../tracker_testing/results/{sub_dir}.txt',
                        'a+',encoding='utf-8') as f:
                        f.write(out_str)


            for box in boxes:
                b_0 = box[0]-box[2]/2
                b_1 = box[1]-box[3]/2
                b_2 = box[0]+box[2]/2
                b_3 = box[1]+box[3]/2
                draw_bounding_box_on_image(im_orig, b_1, b_0, b_3, b_2, use_normalized_coordinates=True)

        if args.view_tracking:
            cv2.imshow("Tracking", np.array(im))#cv2.resize(np.array(im),(1280,800)))
            out.write(np.array(im))

        if args.view_gt:
            cv2.imshow("Ground truth", np.array(im_orig))#cv2.resize(np.array(im_orig),(1280,800)))

        cv2.waitKey(1)
        i += 1
        if FEED == -1:
            with open("../COMM", "w", encoding="utf-8") as f:
                f.write(str(0))

            break

except KeyboardInterrupt:
    with open("../COMM", "w", encoding="utf-8") as f:
        f.write(str(0))
    os.remove(f'../tracker_testing/results/{sub_dir}.txt')

#EOF
