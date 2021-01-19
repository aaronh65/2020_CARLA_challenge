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
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
#export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg

export PROJECT_ROOT=/home/$NAME/workspace/carla/2020_CARLA_challenge
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/scenario_runner


export DEBUG_CHALLENGE=0 # DO NOT MODIFY
export HAS_DISPLAY=1

# leaderboard and agent config
export AGENT=$1
export TEAM_AGENT=$PROJECT_ROOT/leaderboard/team_code/${AGENT}.py
export TEAM_CONFIG=$2
export ROUTE_PATH=$3
export REPETITIONS=$4

#export TEAM_CONFIG=$PROJECT_ROOT/leaderboard/config/${CONFIG}
CHECKPOINT_ENDPOINT="$BASE_SAVE_PATH/logs/${ROUTE_NAME}.txt"



#python leaderboard/leaderboard/leaderboard_evaluator.py \

fil-profile run leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=$PROJECT_ROOT/leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTE_PATH} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${WORLD_PORT} \
--debug=${DEBUG_CHALLENGE} \
--trafficManagerPort=${TM_PORT} \
--repetitions=${REPETITIONS}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

