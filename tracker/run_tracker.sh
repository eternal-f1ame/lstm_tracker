ROOT_PATH=$1
CODE_HOME=$ROOT_PATH
PYTHONPATH=PYTHONPATH:$CODE_HOME
export PYTHONPATH
python run_tracker.py $2 $3 --view_tracking $4 --view_gt $5 --save_tracking $6