# CNN-LSTM Based Security Robot for Violence Detection

## Overview
This project presents an autonomous security robot designed to detect violent behavior in unmonitored alley-like environments. The system integrates computer vision, deep learning, and robotic navigation to provide real-time monitoring and alerting.

A CNN-LSTM model is used for violence detection, where spatial features are extracted from video frames using a Convolutional Neural Network (CNN) and temporal patterns are analyzed using a Long Short-Term Memory (LSTM) network. The robot navigates autonomously using ultrasonic sensors and streams video to an external laptop for model inference.

## Key Features
- Autonomous forward navigation with obstacle avoidance using ultrasonic sensors
- Real-time violence detection using CNN-LSTM architecture
- Live video streaming from Raspberry Pi to laptop
- Immediate response upon detection:
  - Robot stops movement
  - Buzzer alert activation
  - Telegram notification with image, timestamp, and GPS coordinates
- Resume navigation after alert handling

## System Architecture
- Raspberry Pi: Controls motors, ultrasonic sensors, camera streaming, buzzer, and GPS
- Laptop: Runs CNN-LSTM model for violence detection and handles notifications
- Communication: UDP video streaming (FFmpeg) and TCP sockets for control signals

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- CNN-LSTM Deep Learning Model
- Raspberry Pi GPIO
- Ultrasonic Sensors
- FFmpeg (video streaming)
- Telegram Bot API
- GPS Module

## Dataset
The violence detection model was trained using publicly available datasets containing violent and non-violent video clips, including movie scenes and hockey fight datasets.

## Project Outcome
The system successfully demonstrates real-time violence detection integrated with an autonomous robotic platform. It highlights the feasibility of combining deep learning with mobile robotics for proactive surveillance applications.

## Limitations
- GPS accuracy may be limited in indoor environments
- Navigation is forward-based without global mapping
- Model inference depends on network stability between robot and laptop

## Future Improvements
- Integration of ROS for advanced navigation and mapping
- Deployment of the model on edge devices
- Improved localization using additional sensors
- Expansion to multi-directional patrol routes

## Author
Afiqah Zakirah Adnan
