# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:22:56 2021

@author: gauri
"""

import cv2
import streamlit as st 
cascPath1 = "haarcascade_frontalface_default.xml"
cascPath2 =" haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath1)
eyeCascade = cv2.CascadeClassifier(cascPath2)
st.title("live")
run1 = st.checkbox('run camera for face ')
run2 = st.checkbox('run camera for eyes ')
frame_window = st.image([])
camera = cv2.VideoCapture(0)
while run1:
    # Capture frame-by-frame
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
while run2:
        
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    eye = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in eye:
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
   
    frame_window.image(frame)
else:
    st.write('live ended')