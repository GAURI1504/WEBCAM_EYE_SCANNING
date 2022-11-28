# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 11:22:46 2021

@author: gauri
"""

import cv2
import streamlit as st 
cascPath1 = "haarcascade_frontalface_default.xml"
cascPath2 = "haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath1)
eyeCascade = cv2.CascadeClassifier(cascPath2)
st.title("live")
run = st.checkbox('run camera for face ')
run = st.checkbox('run camera for eyes ')
frame_window = st.image([])
video_capture = cv2.VideoCapture(0)
while run:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 1255, 0), 2)

    eye = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 1255, 0), 2)

    # Display the resulting frame
    frame_window.image(frame)
else:
    st.write('live ended')