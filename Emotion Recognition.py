#!/usr/bin/env python
# coding: utf-8

# # Emotion Recognition Deep Learning Python

# Deteksi Emosi secara Real-Time dengan Deep Learning Python

# Macam-Macam Ekspresi Wajah Manusia

# <img src = "ekspresi.png" style="width:600px;height:320">

# # Install Library yang Dibutuhkan

# 1. OpenCV --> pip install opencv-python
# 2. OpenCV Contrib --> pip install opencv-contrib-python
# 3. Keras --> pip install keras
# 4. Tensorflow --> pip install tensorflow
# 5. Imutils --> pip install imutils

# # Code Program Emotion Detection

# ## Masukan Library yang di butuhkan

# In[1]:


import numpy as np
import cv2
import time
from keras_preprocessing import image


# ## Tambahkan File Haarcascade Frontal Face

# Untuk mendeteksi bagian wajah bagian depan

# In[2]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# ## Tambahkan Script Untuk Capture Video

# In[3]:


cap = cv2.VideoCapture(0) # 0 --> Web Cam Internal


# ## Masukan Model Hasil Training

# Model yang berisikan pola wajah untuk identifikasi ekspresi wajah

# In[4]:


from tensorflow.keras.models import model_from_json
model = model_from_json(open('facial_expression_model_structure.json').read())
model.load_weights('facial_expression_model_weights.h5') 


# ## Masukan Jenis-Jenis Emosi

# In[5]:


emotions = ('Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut','Biasa')


# ## Lakukan Looping untuk Mendapatkan Data Secara Real-Time

# In[6]:


while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, (48, 48))
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('img', img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
        break


# ## Menutup Semua Window yang Aktif

# In[7]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# <img src = "thank-you.png" style="align=center">
