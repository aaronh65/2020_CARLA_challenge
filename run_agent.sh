#!/bin/bash

# running on my local machine vs CMU cluster
export NAME=aaron
source /home/$NAME/anaconda3/etc/profile.d/conda.sh
#conda activate lblbc
#conda activate lbrl
conda activate ${CONDA_ENV}

# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$NAME/anaconda3/lib
export CARLA_ROOT=/home/$NAME/workspace/carla/CARLA_0.9.10.1
#export CARLA_ROOT=/home/$NAME/workspace/carla/CARLA_0.9.11
export LB_ROOT=/home/$NAME/workspace/carla/2020_CARLA_challenge
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
#export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export DEBUG_CHALLENGE=0 # DO NOT MODIFY
export HAS_DISPLAY=1


# leaderboard and agent config
export AGENT=$1
export TEAM_AGENT=$LB_ROOT/leaderboard/team_code/${AGENT}.py
export TEAM_CONFIG=$2
export ROUTE_PATH=$3
export REPETITIONS=$4

#export TEAM_CONFIG=$LB_ROOT/leaderboard/config/${CONFIG}
CHECKPOINT_ENDPOINT="$BASE_SAVE_PATH/logs/${ROUTE_NAME}.txt"

python leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTE_PATH} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${WORLD_PORT} \
--trafficManagerPort=${TM_PORT} \
--debug=${DEBUG_CHALLENGE} \
--repetitions=${REPETITIONS}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

