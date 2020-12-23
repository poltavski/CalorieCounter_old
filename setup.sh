#!/bin/bash  

python3 -m venv $(pwd)/.food_classification;

# source $(pwd)/.food_classification/bin/activate;
. $(pwd)/.food_classification/bin/activate;

pip install -r requirements.txt

# install tf model server
apt install curl
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
apt-get update
apt-get install tensorflow-model-server
tensorflow_model_server --version

bash run_TFserving.sh
