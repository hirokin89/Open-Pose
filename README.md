# Open-Pose

This repository is for executing pose estimation utilizing open-pose.

# Language

Python 3.7.4

# Require

Pytorch 

openCV for Python 

CUDA

# How to execute

- you must do clone this repository.

`git clone https://github.com/hirokin89/Open-Pose/`

- you must put a test image or video to the `/data` directory.

- you must download trained model from the following website.

`https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?dl=0`

- Please put the trained model on `/weight` directory.

- execute the program

`python test_openpose.py`

# Reference
小川雄太郎: "つくりながら学ぶ！ Pytorchによる発展ディープラーニング", 2019. 

`https://github.com/YutaroOgawa/`
