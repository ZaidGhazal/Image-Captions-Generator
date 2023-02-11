#!/bin/bash
set -e
ENV_PATH=$HOME/user_environments/image-caption-env
BASEDIR=$(dirname "$0")
REQUIERMENTS_PATH=$BASEDIR/requirements.txt
MAIN_PATH=$BASEDIR/main.py
echo $REQUIERMENTS_PATH
if [ ! -d $ENV_PATH ]
then

  echo "--> 1. Creating the Virtual Environment: image-caption-env ..."
  python3 -m venv $ENV_PATH
  echo "SUCCESS: Virtual Environment Created on: $ENV_PATH"

  echo "--> 2. Activating the Created Environment..."
  source $ENV_PATH/bin/activate
  echo "SUCCESS: Virtual Environment Activated"

  echo "--> 3. Installing Dependencies..."
  pip install --upgrade pip
  python3 -m pip install -r "$REQUIERMENTS_PATH"
  echo "SUCCESS: Dependencies Installed"
else

  echo "--> Activating the Created Environment..."
  source $ENV_PATH/bin/activate
  echo "SUCCESS: Virtual Environment Activated"

  echo "--> Check Dependencies..."
  pip install --upgrade pip
  python3 -m pip install -r "$REQUIERMENTS_PATH"
  echo "SUCCESS: Dependencies Installed"
fi
echo "--> Running App..."
cd "$BASEDIR"
streamlit run "$MAIN_PATH" -- --disable_training 0
