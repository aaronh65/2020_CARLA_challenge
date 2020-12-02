#!/bin/bash

source /home/aaron/.bashrc
source /home/aaron/anaconda3/etc/profile.d/conda.sh
conda activate leaderboard

# Python env variables so the subdirectories can find each other

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aaron/anaconda3/lib
export CARLA_ROOT=/home/aaron/workspace/carla/CARLA_0.9.10.1
export LBC_ROOT=/home/aaron/workspace/carla/2020_CARLA_challenge
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
#export ROUTES=$LBC_ROOT/leaderboard/data/routes_devtest/route_01.xml
export SPLIT=training
export ROUTE_NUM=route_18
export ROUTES=$LBC_ROOT/leaderboard/data/routes_${SPLIT}/${ROUTE_NUM}.xml
export LOGDIR=$LBC_ROOT/leaderboard/data/logs/image_agent/${SPLIT}
export SAVE_IMG_PATH=${LOGDIR}/images/${ROUTE_NUM}
if [ ! -d $SAVE_IMG_PATH ]; then
	mkdir -p $SAVE_IMG_PATH
fi
export LOGDIR=$SAVE_IMG_PATH
echo $SAVE_IMG_PATH
export DEBUG_CHALLENGE=0
export HAS_DISPLAY=1
export PORT=2000

# LBC agent config
export TEAM_AGENT=$LBC_ROOT/leaderboard/team_code/image_agent.py
export TEAM_CONFIG=$LBC_ROOT/leaderboard/data/image_model.ckpt
#export TEAM_AGENT=$LBC_ROOT/leaderboard/team_code/auto_pilot.py
#export TEAM_CONFIG=$LBC_ROOT/leaderboard/data/

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
--debug=${DEBUG_CHALLENGE}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

