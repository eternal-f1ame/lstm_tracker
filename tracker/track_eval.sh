ROOT_PATH=$1
CODE_HOME=$ROOT_PATH
PYTHONPATH=PYTHONPATH:$CODE_HOME
export PYTHONPATH
python track_eval.py $2