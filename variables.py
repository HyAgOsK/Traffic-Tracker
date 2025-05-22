import numpy as np
import streamlit as st
from ultralytics import YOLO

SOURCE_VIDEO_PATH = "1_2_5_10_20_kmhplaca.mp4"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 1.0
model_extension = 'tflite'
MODEL_NAME = "<model_dir_file_name>" #tflite, pt, onnx, edgetpu(tflite) 
MODEL_RESOLUTION =  640
ALPHA = 0.5
SPEED_THRESHOLD = 15 # Speed threshold in km/h to save frames
distance = 0


# ******************** ESTRADA DISTNCIA PADRO *******************
SOURCE_MATRIX = np.array([
    [848,93],
    [1069,93],
    [1752,819],
    [172,819],
])

TARGET_MATRIX = np.array([
    [0,0],
    [10,0],
    [10,100],
    [0,100],
])



