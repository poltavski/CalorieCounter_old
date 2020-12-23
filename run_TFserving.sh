#!bin/bash

tensorflow_model_server --model_base_path=$(pwd)/my_image_classifier --rest_api_port=9000 --model_name=ImageClassifier

