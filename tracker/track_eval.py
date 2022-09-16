import motmetrics as mm
import numpy as np
import argparse
from dataset_utils.kitti_datum import KITTIDataset
from dataset_utils.track_datum import TrackObjHandler

parser = argparse.ArgumentParser()
parser.add_argument(
    'eval_dir', action='store' , type=str,
    help='Directory containing all the frames to use'
    )
eval_dir = parser.parse_args().eval_dir
seq_eval = int(eval_dir)

BASE_PATH = ".."

dataset = KITTIDataset(root_path="{}/data/KITTI_tracking/data_tracking_image_2/training".format(BASE_PATH))
# dataset = MOTDataset(root_path="{}/data/MOT16/train".format(BASE_PATH))

track_seq = TrackObjHandler("{}/tracker_testing/results".format(BASE_PATH), eval_dir)
gt_seq = dataset.sequences[seq_eval]

acc = mm.MOTAccumulator(auto_id=True)

for i, gt in enumerate(gt_seq.datums()):

    if i == 0:
        continue

    i_w, i_h = gt.image.size
    gt_bb = np.array([np.array([x.x_min / i_w, x.y_min / i_h,
                    (x.x_max - x.x_min) / i_w, (x.y_max - x.y_min) / i_h]) for x in gt.objects])

    track = track_seq.tracks[int(gt.img_id[:-4])]
    t_bb = np.array(
        [np.array([x.x_min, x.y_min, x.x_max - x.x_min, x.y_max - x.y_min]) for x in track])

    gt_ids = np.array([x.track for x in gt.objects])
    track_ids = np.array([x.track for x in track])
    iou_dist = mm.distances.iou_matrix(gt_bb, t_bb, max_iou=1)  # x_min, y_min, W, H
    acc.update(gt_ids, track_ids, iou_dist)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'mostly_tracked', 'mostly_lost','partially_tracked',
                                   'num_switches', 'num_matches', 'num_objects','precision','recall','idp','idr'], name='acc')
pretty_summary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(pretty_summary)
