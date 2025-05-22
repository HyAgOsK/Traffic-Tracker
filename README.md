# 🚦 Traffic Tracker - Traffic Violation Detection and Monitoring System

This project is an interactive application built with **Streamlit** that uses computer vision models (YOLOv8) for **vehicle detection and tracking**, **speed analysis**, **distance calculation**, and **traffic violation detection**, such as **speeding** and **collision risk**. The system can also extract license plates using OCR and send data to servers via **MQTT**.

---

## 📌 Features

- **Upload YOLOv8 models** (`.pt`, `.tflite`, `.onnx`)
- **Custom video input:** sample video, webcam, or uploaded file
- **Vehicle tracking with ByteTrack**
- **Speed calculation in km/h using Exponential Moving Average (EMA)**
- **Traffic violation detection:**
  - Speeding
  - Collision risk (vehicles too close)
- **Automatic license plate extraction and multimodal context (Gemini)**
- **Real-time data visualization:**
  - Vehicle count
  - Average speed
  - Number of violations
  - Distance between vehicles
- **Data publishing via MQTT**

---

## 🧠 Technologies and Libraries Used

| Category                   | Technologies/Libraries                             |
|---------------------------|-----------------------------------------------------|
| **UI Framework**          | [Streamlit](https://streamlit.io/)                 |
| **Object Detection**      | [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics) |
| **Tracking**              | [ByteTrack](https://github.com/ifzhang/ByteTrack)  |
| **Geometric Transformations** | `ViewTransformer` using homography           |
| **Speed Analysis**        | EMA (Exponential Moving Average)                   |
| **OCR (License Plate)**   | [Gemini](https://aistudio.google.com/)             |
| **Data Publishing**       | MQTT Publisher                                     |
| **Visualization**         | `matplotlib`, `pandas`, `streamlit charts`         |

---

## ⚙️ Project Structure

```
traffic_tracker/
├── utils/
│   ├── helper.py            # Helper functions (e.g., email sending)
│   ├── constants.py         # Global constants (limits, paths)
├── functions.py             # General functions (e.g., violation log)
├── transformerPoints.py     # Perspective transformation using homography
├── mqtt_publisher.py        # MQTT data publishing
├── variables.py             # Adjustable system parameters
├── main.py                  # Main script
├── README.md                # Project documentation
```

---

## 🎯 How It Works

### 1. Detection and Tracking
- The system detects vehicles using YOLOv8.
- Each vehicle is tracked with a unique ID via **ByteTrack**.

### 2. Speed and Distance Analysis
- Speed is calculated based on position changes in the transformed plane (homography).
- **EMA** smooths fluctuations for more stable output.
- The distance between vehicles is analyzed to predict risks.

### 3. Violation Detection
- **Speeding:** if the speed exceeds `SPEED_THRESHOLD`.
- **Collision risk:** if the distance between moving vehicles is less than the safe minimum.
- When a violation is detected:
  - Captures and saves an image of the vehicle
  - Extracts the license plate using Gemini
  - Generates a violation report
  - Publishes it via MQTT to an external server

---

## 📊 Interface

- Real-time video visualization with annotations:
  - Vehicle ID
  - Speed (km/h)
  - Critical distance (if applicable)
- Four dynamic charts:
  - **Vehicle count**
  - **Average speed**
  - **Number of violations**
  - **Average vehicle distance**

---

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/traffic-tracker.git
   cd traffic-tracker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

🛠 Requirements:
Python 3.8+

Libraries: streamlit, opencv, numpy, ultralytics, doctr, paho-mqtt, etc.

📩 Contact:
For questions, suggestions, or contributions, feel free to contact:

📧 Email: hyago.silva@mtel.inatel.br
