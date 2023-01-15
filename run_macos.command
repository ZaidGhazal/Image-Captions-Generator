#!/bin/bash
set -e
ENV_PATH=$HOME/user_environments/coco_env
BASEDIR=$(dirname "$0")
REQUIERMENTS_PATH=$BASEDIR/requirements.txt
MAIN_PATH=$BASEDIR/main.py
echo $REQUIERMENTS_PATH
if [ ! -d $ENV_PATH ]
then

  echo "--> 1. Creating the Virtual Environment: my_env ..."
  python3 -m venv $ENV_PATH
  echo "SUCESS: Virtual Environment Created on: $ENV_PATH"

  echo "--> 2. Activating the Created Environment..."
  source $ENV_PATH/bin/activate
  echo "SUCESS: Virtual Environment Activated"

  echo "--> 3. Installing Dependencies..."
  pip install --upgrade pip
  python3 -m pip install -r "$REQUIERMENTS_PATH"
  echo "SUCESS: Dependencies Installed"
else

  echo "--> Activating the Created Environment..."
  source $ENV_PATH/bin/activate
  echo "SUCESS: Virtual Environment Activated"

  echo "--> Check Dependencies..."
  pip install --upgrade pip
  python3 -m pip install -r "$REQUIERMENTS_PATH"
  echo "SUCESS: Dependencies Installed"
fi
echo "--> Running App..."
python3 "$MAIN_PATH"

