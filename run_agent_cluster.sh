#!/bin/bash

source /home/aaronhua/.bashrc
conda activate lb

# Python env variables so the subdirectories can find each other
export CUDA_VISIBLE_DEVICES=$1
export PORT=$2
export ROUTES=$3
export LOGDIR=$4
export TM_PORT=$5
export REPETITIONS=$8

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aaronhua/anaconda3/lib
export CARLA_ROOT=/home/aaronhua/CARLA_0.9.10.1
export LBC_ROOT=/home/aaronhua/2020_CARLA_challenge
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export HAS_DISPLAY=0
export DEBUG_CHALLENGE=0

# LBC agent config
export TEAM_AGENT=$LBC_ROOT/leaderboard/team_code/${6}.py
export TEAM_CONFIG=$LBC_ROOT/leaderboard/data/$7

# leaderboard config


if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$LOGDIR/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$LOGDIR/$(basename $ROUTES .xml).txt"
fi

python leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--debug=${DEBUG_CHALLENGE} \
--repetitions=${REPETITIONS}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

