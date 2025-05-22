import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
from tqdm import tqdm
import math
import time
import matplotlib.pyplot as plt
from utils.helper import send_email
from utils.constants import *
from deepsparse import Pipeline
from deepsparse.yolo.schemas import YOLOInput
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from inference_sdk import InferenceHTTPClient
from utils.constants import (SENDER_ADDRESS, PORT, SMTP_SERVER_ADDRESS, SENDER_PASSWORD)
from variables import *
import pandas as pd
from trasformerPoints import ViewTransformer
from functions import *
from mqtt_publisher import Servidor

ocr_model = ocr_predictor(pretrained=True)

st.set_page_config(layout="wide")
st.title("Traffic Tracker")

# Inicializa MQTT Publisher
mqtt_server = Servidor()

uploaded_model_file = st.sidebar.file_uploader("Upload the Yolov8 model", type=["pt", "tflite", "onnx"])

if uploaded_model_file is not None:
    model_extension = uploaded_model_file.name.split('.')[-1]
    if model_extension == 'pt':
        with open("uploaded_model.pt", "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        MODEL_NAME = "uploaded_model.pt"
        MODEL_RESOLUTION = 640
        model = YOLO(MODEL_NAME)
    elif model_extension == 'tflite':
        MODEL_RESOLUTION = 640
        with open("uploaded_model.tflite", "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        MODEL_NAME = "uploaded_model.tflite"
        model = YOLO(MODEL_NAME)
    elif model_extension == 'onnx':
        MODEL_RESOLUTION = 640
        with open("uploaded_model.onnx", "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        MODEL_NAME = "uploaded_model.onnx"
        model = Pipeline.create(task='yolov8', model_path="uploaded_model.onnx",)
else:
    model = YOLO(MODEL_NAME)

CONFIDENCE_THRESHOLD = st.sidebar.slider("Set the model confidence threshold", 0.0, 1.0, 0.2)
previous_ema_dist_obj = None
source_type = st.sidebar.selectbox("Choose the source (video/webcam)", ["default video", "upload video", "webcam"])

if source_type == "default video":
    video_source = SOURCE_VIDEO_PATH
    video_info = sv.VideoInfo.from_video_path(video_path=video_source)
    frame_generator = sv.get_video_frames_generator(source_path=video_source)
elif source_type == "upload video":
    uploaded_file = st.sidebar.file_uploader("upload video here", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        video_source = uploaded_file.name
        with open(video_source, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_info = sv.VideoInfo.from_video_path(video_path=video_source)
        frame_generator = sv.get_video_frames_generator(source_path=video_source)
    else:
        st.stop()
if source_type == "Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam access error")
        st.stop()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info = sv.VideoInfo(fps=fps, width=width, height=height)
    frame_generator = (cap.read()[1] for _ in iter(int, 1))
else:
    video_info = sv.VideoInfo.from_video_path(video_path=video_source)
    frame_generator = sv.get_video_frames_generator(source_path=video_source)

view_transformer = ViewTransformer(source=SOURCE_MATRIX, target=TARGET_MATRIX)
previous_ema = None
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD
)
text_scale = min(video_info.resolution_wh) * 1e-3
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=2,
    text_position=sv.Position.BOTTOM_CENTER
)
polygon_zone = sv.PolygonZone(polygon=SOURCE_MATRIX)
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

left_column, right_column = st.columns(2)
video_placeholder = left_column.empty()
chart_placeholder = right_column.empty()

timestamps = []
object_counts = []
speed_view_counts = []
distances = []

# Placeholder para o grÃ¡fico no Streamlit
chart_placeholder = st.empty()


col1, col2, col3, col4 = st.columns(4)

# Placeholders para os grÃ¡ficos em cada coluna
with col1:
    chart_placeholder_flux = st.empty()

with col2:
    chart_placeholder_speed = st.empty()

with col3:
    chart_placeholder_infra = st.empty()

with col4:
    chart_placeholder_dist = st.empty()

def update_chart():
    # Verifique se o tamanho dos timestamps e object_counts Ã© igual
    if len(timestamps) == len(object_counts) and len(timestamps) == len(infractions_counts):
        # Crie um novo DataFrame com as listas atuais
        timestamps_rounded = [round(t, 2) for t in timestamps]
        df_1 = pd.DataFrame({
            "timestamps": timestamps_rounded,
            "object_counts": object_counts
        })

        df_2 = pd.DataFrame({
            "timestamps": timestamps_rounded,
            "speed_view_counts": speed_view_counts
        })

        df_3 = pd.DataFrame({
            "timestamps": timestamps_rounded,
            "infractions_counts": infractions_counts
        })
        
        df_4 = pd.DataFrame({
            'timestamps': timestamps_rounded,
            'distances': 0
        })

        # Exiba o grÃ¡fico usando o DataFrame
        chart_placeholder_flux.line_chart(df_1.set_index("timestamps"), color='#f00', x_label='Tempo(min)', y_label='Fluxo de objetos')
        chart_placeholder_speed.line_chart(df_2.set_index("timestamps"), color='#f00', x_label='Tempo(min)', y_label='Velocidade (Km/h)')
        chart_placeholder_infra.line_chart(df_3.set_index("timestamps")["infractions_counts"], color='#f00', x_label='Tempo(min)', y_label='NÃºmero de InfraÃ§Ãµes')
        chart_placeholder_dist.line_chart(df_4.set_index("timestamps"), color='#f00', x_label='Tempo(min)', y_label='DistÃ¢ncia entre os objetos')
        
        # Salva no MongoDB via MQTT se os dados existirem 
        #if object_counts and speed_view_counts and infractions_counts and distances:
            
       
    
    else:
        pass

time_window = 2.0
object_count_window = deque(maxlen=int(video_info.fps * time_window))
time_start_window = time.time()
frame_count = 0
frame_count_distance = 0
frame_count_speed = 0
start_time = time.time()
previous_ema_dist = None
speed_view = [0] 
infractions_counts = []
processed_ids = set() 
frames_captured = defaultdict(int)
houve_infracao = False
placa = "sem infracao"
image_path = "nenhuma imagem salva"
type_infraction = placa
tracker_id = "nenhum objeto detectado"
ultima_infracao_enviada = None

# Checkbox para habilitar/desabilitar a exibiÃ§Ã£o do vÃ­deo
show_video = st.sidebar.checkbox("Show video", value=True)

for frame in tqdm(frame_generator, total=video_info.total_frames):
    frame_count += 1
    frame_count_distance += 1
    frame_count_speed += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    if model_extension == 'onnx':
        input_data = YOLOInput(images=[frame])
        pipeline_outputs = model(images=[frame])
        detections = sv.Detections.from_deepsparse(pipeline_outputs)
    else:
        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[polygon_zone.trigger(detections)]
    detections = detections.with_nms(IOU_THRESHOLD)
    detections = byte_track.update_with_detections(detections=detections)
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_points = view_transformer.transformPointsHomography(points=points)

    for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
        coordinates[tracker_id].append(y)

    labels = []
    for tracker_id in detections.tracker_id:
        if tracker_id not in coordinates or len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
            speed_view.append(0)
        else:
            coordinate_start = coordinates[tracker_id][0]
            coordinate_end = coordinates[tracker_id][-1]
            distance_obj = abs(coordinate_end - coordinate_start)
            time_interval = len(coordinates[tracker_id]) / video_info.fps

            if previous_ema_dist is None:
                previous_ema_dist = distance_obj
                ema_speed_d = distance_obj
            else:
                ema_speed_d = calculate_ema(previous_ema_dist, distance_obj, 1)
                previous_ema_dist = ema_speed_d

            speed = distance_obj / time_interval * 3.6
            
            if previous_ema is None:
                previous_ema = speed
                ema_speed = speed
                speed_view.append(0)
            else:
                ema_speed = calculate_ema(previous_ema, speed, ALPHA)
                previous_ema = ema_speed
                speed_view.append(speed)
                
                for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
                    if tracker_id not in processed_ids:
                            if distance <= 6.0 and distance!= 0 and frames_captured[tracker_id] < 1 and frame_count_distance>=5:
                                    c = "Risk of collision"
                                    frame_count_distance = 0
                                    houve_infracao = True
                                    x1, y1, x2, y2 = map(int, bbox)
                                    vehicle_crop = frame[y1:y2, x1:x2]
                                    image_path = f"id_{tracker_id}.jpg"
                                    cv2.imwrite(image_path, vehicle_crop)
                                    
                                    frames_captured[tracker_id] += 1  
                                    processed_ids.add(tracker_id)
                                        
                                    placa = genai_ocr(image_path)
                                    
                                                                      
                                    infraction_details = (
                                                f"Data e Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                                f"Type of infraction {type_infraction}!\n"
                                                f"Distancia entre objetos: {distance:.2f} metros\n"
                                                f"Velocidade do objeto: {speed:.2f} km/h\n\n"
                                                f"Placa detectada: {placa}\n\n"
                                                f"Imagem salva em: {image_path}\n"
                                                "---------------------------------------"
                                    )
                                    log_infraction(infraction_details)

                            elif previous_ema >= SPEED_THRESHOLD and frames_captured[tracker_id] < 1 and frame_count_speed >= 5:
                                    type_infraction = "speed limit exceeded"
                                    frame_count_speed = 0
                                    houve_infracao = True
                                    x1, y1, x2, y2 = map(int, bbox)
                                    vehicle_crop = frame[y1:y2, x1:x2]
                                    image_path = f"id_{tracker_id}.jpg"
                                    cv2.imwrite(image_path, vehicle_crop)
                                    
                                    frames_captured[tracker_id] += 1  
                                    processed_ids.add(tracker_id)
                                        
                                    
                                    placa = genai_ocr(image_path)
                                                                         
                                    infraction_details = (
                                                f"Data e Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                                f"tipo de infraÃ§ao {type_infraction}!\n"
                                                f"Distancia entre objetos: {distance:.2f} metros\n"
                                                f"Velocidade do objeto: {speed:.2f} km/h\n"
                                                f"Placa detectada: {placa}\n"
                                                f"Imagem salva em: {image_path}\n"
                                                "---------------------------------------"
                                    )
                                    log_infraction(infraction_details)

                            #placa = ''
                            if previous_ema_dist <= 0.9 and time_interval <= 1:
                                labels.append(f"#{tracker_id} {0} km/h")
                            else:
                                labels.append(f"#{tracker_id} {int(ema_speed)} km/h")
                            

    num_detections = len(detections)
    num_labels = len(labels)
    if num_detections != num_labels:
        continue

    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    for i in range(len(transformed_points) - 1):
        for j in range(i + 1, len(transformed_points)):
            point1 = tuple(points[i])
            point2 = tuple(points[j])
            distance = calculate_euclidean_distance(transformed_points[i], transformed_points[j])

                        
            if previous_ema_dist_obj is None:
                previous_ema_dist_obj = distance
                ema_speed_dist_obj = distance
                distance = 0
            else:
                ema_speed_dist_obj = calculate_ema(previous_ema_dist_obj, distance, ALPHA)
                previous_ema_dist_obj = ema_speed_dist_obj

            annotated_frame = draw_distance_line(annotated_frame, point1, point2, previous_ema_dist_obj)
    
     
     
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    print('fps',fps)
    if show_video:
        video_placeholder.image(annotated_frame, channels="BGR")

    object_count_window.append(len(detections))
    current_time = time.time()
    if current_time - time_start_window >= time_window:
            
        timestamps.append((current_time - start_time) / 60)
        object_counts.append(len(detections))
        speed_view_counts.append(speed_view[-1])
        distances.append(distance)
        if houve_infracao:
            infractions_counts.append(1)
            #mqtt_server.publish_result(int(len(detections)), round(speed_view[-1],2), 1, round(distance,2), time.strftime('%Y-%m-%d %H:%M:%S'), str(placa), str(image_path), str(type_infraction) , str(tracker_id))
            nova_infracao = (
                int(len(detections)),
                round(speed_view[-1], 2),
                1,
                round(distance, 2),
                time.strftime('%Y-%m-%d %H:%M:%S'),
                str(placa),
                str(image_path),
                str(type_infraction),
                str(tracker_id)
            )

            
            if nova_infracao != ultima_infracao_enviada:
                mqtt_server.publish_result(*nova_infracao)
                ultima_infracao_enviada = nova_infracao
            else:
                print("Infracao repetida ignorada.")
                    
        else:
            infractions_counts.append(0)
            nova_infracao = (
                int(len(detections)),
                round(speed_view[-1], 2),
                1,
                round(distance, 2),
                time.strftime('%Y-%m-%d %H:%M:%S'),
                str(placa),
                str(image_path),
                str(type_infraction),
                str(tracker_id)
            )

            
            if nova_infracao != ultima_infracao_enviada:
                mqtt_server.publish_result(*nova_infracao)
                ultima_infracao_enviada = nova_infracao
            else:
                print("infracao repetida ignorada.")

        houve_infracao = False
        update_chart()
        time_start_window = current_time
        object_count_window.clear()

st.write("Processamento de vÃ­deo concluÃ­do!")

