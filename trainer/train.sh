ROOT_PATH=$1
CODE_HOME=$ROOT_PATH
PYTHONPATH=PYTHONPATH:$CODE_HOME
export PYTHONPATH
python train.py \
    --data_path1 $CODE_HOME/data/MOTA/ \
    --data_path2 $CODE_HOME/data/MOTA/ \
    --job_dir $CODE_HOME/models/exp04 \
    --lr 0.001 \
    --batch 128 \
    --epochs 10000 \
    --eval_int 300
