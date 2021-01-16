#!/bin/bash

# running on my local machine vs CMU cluster
export NAME=aaron
source /home/$NAME/anaconda3/etc/profile.d/conda.sh
#conda activate lblbc
conda activate lbrl
#conda activate ${CONDA_ENV}

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

# leaderboard and agent config
#export ROUTES=$PROJECT_ROOT/leaderboard/data/routes_devtest.xml
export ROUTES=$PROJECT_ROOT/leaderboard/data/routes_training.xml
export SCENARIOS=$PROJECT_ROOT/leaderboard/data/no_traffic_scenarios.json
#export SCENARIOS=$PROJECT_ROOT/leaderboard/data/all_towns_traffic_scenarios_public.json
export REPETITIONS=1

#export TEAM_CONFIG=$PROJECT_ROOT/leaderboard/config/${CONFIG}
CHECKPOINT_ENDPOINT="$BASE_SAVE_PATH/logs/${ROUTE_NAME}.txt"

#python $PROJECT_ROOT/leaderboard/team_code/rl/generate_dense_waypoints.py

python $PROJECT_ROOT/leaderboard/team_code/rl/trainer.py \
	--routes=$ROUTES \
	--scenarios=$SCENARIOS \
	--repetitions=$REPETITIONS

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

