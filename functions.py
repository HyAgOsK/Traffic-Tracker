import math
import cv2
import numpy as np
import requests
import base64
import requests
import google.generativeai as genai
import base64
import io
import json
import mimetypes
import pathlib
import pprint
import requests
import PIL.Image
import IPython.display
from IPython.display import Markdown

def calculate_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def draw_distance_line(scene, point1, point2, distance):
    point1 = tuple(np.round(point1).astype(int))
    point2 = tuple(np.round(point2).astype(int))
    cv2.line(scene, point1, point2, (0, 255, 0), 2)
    text_position = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(scene, f"{distance:.2f} m", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4)
    return scene

def calculate_ema(previous_ema, current_value, alpha):
    return alpha * current_value + (1 - alpha) * previous_ema

def log_infraction(details: str, log_file="logs.txt"):
    with open(log_file, "a") as log:
        log.write(details + "\n")


def genai_ocr(img_path):
    genai.configure(api_key='<API_KEY_MODEL_GEMINI>') 

    # Example usage with the image file
    model = "gemini-2.0-flash"
    contents = [{'parts': [{'image': {'image_url': ''}}, {'text': "Voce e um radar que detecta placas de veiculos e analisa o contexto da cena. Retorne diretamente os caracteres das placas observadas, seguidos por uma explicacao do que esta acontecendo, incluindo colisoes, presenca de pessoas ou detalhes sobre o ambiente. Nao inclua caracteres especiais na resposta, apenas texto limpo. Mantenha a resposta breve e natural, sem introducoes."}]}]

    # Load the image
    try:
      with open(f'{img_path}', 'rb') as f:
        image_data = f.read()
      mime_type = 'image/png' 
    except FileNotFoundError:
        print(f"Error: {img_path} path not found. Please upload or provide the correct path.")
        exit()


    # Modify the contents to include image data
    contents[0]['parts'][0] = {'inline_data': {'data': image_data, 'mime_type': mime_type}}

    # Call the Gemini API
    gemini = genai.GenerativeModel(model_name=model)

    try:
      response = gemini.generate_content(contents)
      return response.text

    except Exception as e:
      print(f"An error occurred: {e}")


