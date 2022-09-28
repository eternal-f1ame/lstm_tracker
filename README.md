# LSTSM Multi Object Tracking Tracking

## Requirenments

* Anaconda3/Miniconda3

* Tensorflow 2.9
* Pytorch (latest)


```Terminal
pip install -r requirements.txt
```

## Training The LSTM Tracker

### Put all the MOTA format .txt files for training in ```<PATH>/lstm_tracker/data/MOTA/```

* MOTA format => `<frame_id> <track_id> <x_centre> <y_centre> <w> <h> <_obj_class>`

### Open a Terminal and Navigate to the training folder

```Terminal
cd <PATH>/lstm_tracker/trainer/
chmod +x train.sh #(only needed the first time)
train.sh <PATH>/lstm_tracker/
```

## Running an Inference with the Tracker

### Download KITTI Object Tracking Video Sequences and Save all the sequences from 0000 to 0020 in ```<PATH>/lstm_tracker/tracker_testing/data/```

```Terminal
cd <PATH>/lstm_tracker/tracker/
chmod +x run_tracker.sh #(only needed the first time)
run_tracker.sh <PATH>/lstm_tracker/ <PATH>/lstm_tracker/tracker_testing/data/<FOLDER> <INT>
```

* `<INT>` is an integer corresponding to `<FOLDER>` sequence.

* `<FOLDER>` should have frame wise txt file in a sub folder named `labels`

## Real Time Tracking

* Open a Terminal window

```Terminal
cd <PATH>/lstm_tracker/yolo/
python detect.py --project <PATH>/lstm_tracker/tracker_testing/ --source <PATH>/lstm_tracker/tracker_testing/data/<FOLDER> --save-txt
```

* Here `<FOLDER>` is the directory which contains the videos in form of frames.
* Open a new Terminal window

```Terminal
cd <PATH>/lstm_tracker/tracker/
chmod +x run_tracker.sh #(only needed the first time)
run_tracker.sh <PATH>/lstm_tracker/ <PATH>/lstm_tracker/tracker_testing/data/<FOLDER> -1

```

* `<FOLDER>` is same as in the previous command

## Results

* The video with GT boxes is stored in `<PATH>/lstm_tracker/tracker_testing/exp/FOLDER` and GT boxes coordinates are stored in `<PATH>/lstm_tracker/tracker_testing/exp/FOLDER/labels/` (in a frame wise .txt file format)
* Results of MOT are stored in `<PATH>/lstm_tracker/tracker_testing/results/FOLDER.txt`

### Evaluation

* Evaluation for KITTI Test Dataset
* Open a Terminal window

```Terminal
cd <PATH>/lstm_tracker
chmod +x track_eval.sh #(only needed for the first time)
track_eval.sh <PATH>/lstm_tracker/

```

* The Evaluation Result is displayed in the Terminal window
