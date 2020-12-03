#!/bin/bash

# running on my local machine vs CMU cluster
export NAME=aaron
#export NAME=aaronhua
source /home/$NAME/anaconda3/etc/profile.d/conda.sh
conda activate lb


# Python env variables so the subdirectories can find each other
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$NAME/anaconda3/lib
export CARLA_ROOT=/home/$NAME/workspace/carla/CARLA_0.9.10.1
export LBC_ROOT=/home/$NAME/workspace/carla/2020_CARLA_challenge
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export DEBUG_CHALLENGE=0 # DO NOT MODIFY
export HAS_DISPLAY=1


# leaderboard and agent config
export AGENT=$1
export ROUTE=$2
export LOGDIR=$3
export CONFIG=$4
export SAVE_IMAGES=$5
export PORT=2000
#export ROUTES=$LBC_ROOT/leaderboard/data/routes_${SPLIT}/route_${ROUTENUM}.xml
export TEAM_AGENT=$LBC_ROOT/leaderboard/team_code/${AGENT}.py
export TEAM_CONFIG=$LBC_ROOT/leaderboard/config/${CONFIG}


# logging
if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$LOGDIR/logs/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$LOGDIR/logs/$(basename $ROUTES .xml).txt"
fi

python leaderboard/leaderboard/leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTE} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${PORT} \
--debug=${DEBUG_CHALLENGE}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

