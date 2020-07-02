import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import time

from PIL import Image, ImageDraw
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#------------------------------------------------------------------------------
# 画像を読み込む。
#img = cv2.imread("sunglass.jpg")

# グレースケールに変換する。
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2値化する。
#thresh, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

# 輪郭を抽出する。
#contours, hierarchy = cv2.findContours(
#    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#)

# マスクを作成する。
#mask = np.zeros_like(binary)

# 輪郭内部 (透明化しない画素) を255で塗りつぶす。
#cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

# RGBA に変換する。
#rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

# マスクをアルファチャンネルに設定する。
#rgba[..., 3] = mask

# 保存する。
#cv2.imwrite(r"result.png", rgba)
#-----------------------------------------------------------------------------

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# emotions will be displayed on your face from the webcam feed
model.load_weights('date/model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# add
face_cascade = cv2.CascadeClassifier('date/haarcascade_frontalface_alt.xml')
facecasc = cv2.CascadeClassifier('date/haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('date/haarcascade_righteye_2splits.xml')

# mosaic
def mosaic(img, rect, size):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]

    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)

    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2

# gasou
def image(img, rect):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1

    img_face = cv2.resize(img1, (w, h))
    img2 = img.copy()
    img2[y1:y2, x1:x2] = img_face
    return img2

cou0=cou1=cou2=cou3=cou4=cou5=cou6=cou7=0

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    faces = face_cascade.detectMultiScale(gray, 1.5, 3)
    

    for (x, y, w, h) in faces:
        
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        print(maxindex)

        #img1 = cv2.imread('img/sad/5.png')
        #ret, img = cap.read()
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #frame=image(img, (x, y, x+w, y+h))

        if maxindex == 0:
            eye = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye:
                ret, img = cap.read()
                img1 = cv2.imread('img/fearful/2')
                #img1 = rgba
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))
                #ret, img = cap.read()
                #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #frame=mosaic(img, (x, y, x+w, y+h), 10)
            
        elif maxindex == 1:
            eye = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye:
                ret, img = cap.read()
                img1 = cv2.imread('img/fearful/2')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))
 
        elif maxindex ==2:
            eye = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye:
                ret, img = cap.read()
                img1 = cv2.imread('img/fearful/2')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))

        elif maxindex ==3:
            eye = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye:
                ret, img = cap.read()
                img1 = cv2.imread('img/fearful/2')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))       

        elif maxindex == 4:
            eye = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye:
                ret, img = cap.read()
                img1 = cv2.imread('img/fearful/2')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))

        elif maxindex == 5:
            eye = eyes.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eye:
                ret, img = cap.read()
                img1 = cv2.imread('img/fearful/2')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))
  
                #img1 = cv2.imread('img/sad/5.png')
                #ret, img = cap.read()
                #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #frame=image(img, (x, y, x+w, y+h))


    cv2.imshow('Video', cv2.resize(frame,(600,460),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()