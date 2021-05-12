Author: Juneki Hong (junekihong@gmail.com)
Last updated: January 2021


To train:
python3 src/main.py train --model-path-base [model_directory_name]

This will train a paddle model and save it in the specified directory.
Note: You may need to train a few times, as the model might have bad initialization and may not train well. (A dev MSE loss of about 0.05-0.08 is good)

Example:
python3 src/main.py train --model-path-base model


To test:
python3 src/main.py test --model-path-base [saved_model_directory]

This will evaluate a trained model

Example:
python3 src/main.py test --model-path-base model/model_dev\=0.0772/