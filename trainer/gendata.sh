ROOT_PATH=$1
CODE_HOME=$ROOT_PATH
PYTHONPATH=PYTHONPATH:$CODE_HOME
export PYTHONPATH
python data.py \
    --data_path1 $CODE_HOME/data/kitti_tracks_{}.json \
    --data_path2 $CODE_HOME/data/mot_tracks_{}.json \
    --job_dir $CODE_HOME/models/exp04 \
    --lr 0.001 \
    --batch 128 \
    --epochs 1000 \
    --eval_int 300