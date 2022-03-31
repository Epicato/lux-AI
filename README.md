# Lux AI Challenge 🍄 🎮

Course Project 2 for CS410

## Introduction
This is the implementation of Imitation Learning method for team _No face_ in the [Lux AI Challenge](https://www.kaggle.com/competitions/lux-ai-2021/overview) on kaggle.  
We have used U-Net for 3 versions. Besides the base model, we have tried to add Convolutional Block Attention Module to U-Net. Moreover, we let the model predict 3 actions instead of 6 actions, by integrating conditions of moving in 4 directions to 1 direction. We realized it by rotating the observation and action map.

## Train and test for yourself
The training data can be downloaded from the battle records of the 1st team _Toad Brigade_. You can also download different teams' episodes from [Kaggle Datasets](https://www.kaggle.com/datasets).  
For training, you can run train.py.
For testing the agent's performance, please follow the instructions in this [Tutorial](https://www.kaggle.com/code/stonet2000/lux-ai-season-1-jupyter-notebook-tutorial/notebook).  

## Result
Finally our team scored 1436.3(93/1178).  
Our agent trained with U-Net has become a smart game player!  
![]('game.png')
