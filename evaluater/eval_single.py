import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf=tf.compat.v1
tf.logging.set_verbosity(tf.logging.ERROR)
from evaluater.graph_runner import GraphRunner
runner = GraphRunner(base_path='/home/dark/Documents/GitHub/lstm_tracker',experiment='exp04',checkpoint="model.ckpt-3363",model_params={'Eval_IOU': 0.5, 'output_bins': 5, 'hidden layers': [32, 32], 'num_classes': 9, 'num_timesteps': 10})
from trainer.data import kitti_data_gen
from trainer.helpers import bbox_overlap_iou_np, lbtr_to_chw, to_one_hot

from trainer.dataset_info import kitti_classes_reverse

from dataset_utils.kitti_datum import KITTIDataset
import warnings
warnings.filterwarnings("ignore")
BASE_PATH = "/home/dark/Documents/GitHub/lstm_tracker"

gen = kitti_data_gen(path="{}/data/kitti_tracks_{}.json".format(BASE_PATH, "{}"),
                     split='val', testing=True, one_hot_classes=True, anchors=False)

ious = []
pbar = tqdm()

BASE_PATH = "/home/dark/Documents/GitHub/lstm_tracker"
dataset = KITTIDataset(root_path="{}/data/KITTI_tracking/data_tracking_image_2/training".format(BASE_PATH))
seq = dataset.sequences[0]
dat = seq.random_datum()
i_w, i_h = dat.image.size
boxes = lbtr_to_chw(np.array([[x.x_min / i_w, x.y_min / i_h, x.x_max / i_w, x.y_max / i_h,
                               kitti_classes_reverse[x.category]] for x in dat.objects]))
boxes = to_one_hot(boxes, 9)

# boxes = np.array([runner.prepare_inputs_np(x) for x in boxes])

while True:
    try:
        x, y, x_im, y_im = next(gen)
    except StopIteration:
        break

    x = x[5:, :]

    x = runner.prepare_inputs_np(x)
    y_pred = runner.get_predictions(x)

    iou = bbox_overlap_iou_np(np.expand_dims(y[:4], axis=0), np.expand_dims(y_pred[:4], axis=0)).ravel()[0]

    ious.append(iou)

    pbar.update(1)

# with open("{}/data/results/ious.txt".format(BASE_PATH)) as fo:
#     fo.writelines(ious)

# image = ImageBoxes(path=y_im, plt_axis_off=True)
# image.add_from_track(x[0], y, y_pred[0])
# image.show_plt(np.array(image.get_final()))
