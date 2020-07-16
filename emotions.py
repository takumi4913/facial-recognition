import numpy as np
import argparse
import cv2
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from PIL import Image


os.makedirs('emotions',exist_ok=True)

#add
raw_input=input
input_alpha=0.5

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
face_cascade_alt = cv2.CascadeClassifier('date/haarcascade_frontalface_alt.xml')
face_cascade_default = cv2.CascadeClassifier('date/haarcascade_frontalface_default.xml')

# paste_image
def paste_image(img, rect):
    (x_tl, y_tl, x_br, y_br) = rect
    wid = x_br - x_tl
    hei = y_br - y_tl

    #img_face = cv2.resize(input_img, (wid, hei))
    img_face = cv2.resize(add_img, (wid, hei))
    paste_img = img.copy()
    paste_img[y_tl:y_br, x_tl:x_br] = img_face
    return paste_img

# apply_mosaic
def apply_mosaic(mosaic_img):
    small_mosaic_image = cv2.resize(mosaic_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small_mosaic_image, mosaic_img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #add
    cv2.imwrite(r"user.png",frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recognized_faces = face_cascade_default.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    recognized_faces = face_cascade_alt.detectMultiScale(gray, 1.5, 3)
    
    frame = frame .copy()
    frame = apply_mosaic(frame)

    for (x, y, w, h) in recognized_faces:

        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        for i in range(0, 1, 1):
            time.sleep(0.3)
            maxindex = int(np.argmax(prediction))
        #maxindex = int(np.argmax(prediction))
        
        print(maxindex)
            
        if maxindex == 0:
                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
         
        elif maxindex == 1:
                # input_img = cv2.imread('img/disgusted/disgusted.png')
                # ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # frame=paste_image(img, (x, y, x+w, y+h))
                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
        
        elif maxindex ==2:
                # input_img = cv2.imread('img/fearful/fearful.png')
                # ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # frame=paste_image(img, (x, y, x+w, y+h))

                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv2.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
        
        elif maxindex ==3:
                # input_img = cv2.imread('img/happy/happy.png')
                # ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # frame=paste_image(img, (x, y, x+w, y+h))

                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
        
        elif maxindex == 4:
                # input_img = cv2.imread('img/neutral/neutral.png')
                # ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # frame=paste_image(img, (x, y, x+w, y+h))

                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
        
        elif maxindex == 5:
                # input_img = cv2.imread('img/sad/sad.jpg')
                # ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # frame=paste_image(img, (x, y, x+w, y+h))

                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
        
        elif maxindex == 6:
                # input_img = cv2.imread('img/surprised/surprised.png')
                # ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # frame=paste_image(img, (x, y, x+w, y+h))

                input_img = cv2.imread('img/angry/angry.png')
                input_img_2=cv2.imread("user.png")
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #add
                add_img = cv.addWeighted(input_img, 0.5, input_img_2, 0.5, 0.0)
                frame=paste_image(img, (x, y, x+w, y+h))
        
    cv2.imshow('Video', cv2.resize(frame,(600,460),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------
#eyes = cv2.CascadeClassifier('date/haarcascade_righteye_2splits.xml')

# eye = eyes.detectMultiScale(roi_gray)
# import matplotlib.pyplot as plt

# for (ex, ey, ew, eh) in eye:
    # ret, img = cap.read()
    # img1 = cv2.imread('img/fearful/2.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # frame=image(img,((ex+x)-20,(ey+y)+30,(ex+ew+x)-20,(ey+eh+y)+30))

# mosaic
# def mosaic(img, rect, size):
#     (x1, y1, x2, y2) = rect
#     w = x2 - x1
#     h = y2 - y1
#     i_rect = img[y1:y2, x1:x2]

#     i_small = cv2.resize(i_rect, (size, size))
#     i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)

#     img2 = img.copy()
#     img2[y1:y2, x1:x2] = i_mos
#     return img2

#------------------------------------------------------------------------------
#画像を読み込む。
# img=Image.open("sunglass.jpg").convert('RGB').save('sunglass_rgba')

# img = cv2.imread("sunglass_rgba.jpg")

# #グレースケールに変換する。
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #2値化する。
# thresh, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

# #輪郭を抽出する。
# contours, hierarchy = cv2.findContours(
#    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# #マスクを作成する。
# mask = np.zeros_like(binary)

# #輪郭内部 (透明化しない画素) を255で塗りつぶす。
# cv2.drawContours(mask, contours, -1, color=255, thickness=-1)

# #RGBA に変換する。
# rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

# #マスクをアルファチャンネルに設定する。
# rgba[..., 3] = mask

# cv2.imwrite(r"result.png", rgba)
#-----------------------------------------------------------------------------
#import cv2
#alpha = 0.5
#
## 画像取り込み
#src0 = cv2.imread(cv2.samples.findFile('user.png'))
#src = [cv2.imread(cv2.samples.findFile('0.png')),
#        cv2.imread(cv2.samples.findFile('1.png')),
#        cv2.imread(cv2.samples.findFile('2.png')),
#        cv2.imread(cv2.samples.findFile('3.png')),
#        cv2.imread(cv2.samples.findFile('5.png')),
#        cv2.imread(cv2.samples.findFile('6.png')),]
#
## α値を入力
#print('''0.3~0.8の間から入力
#-----------------------
#* Enter alpha [0.0-1.0]: ''')
#input_alpha = float(input().strip())
#if 0 <= alpha <= 1:
#    alpha = input_alpha
#
## 特定の表情色・imageと合成
#for i in range(6):
#    if src0 is None:
#        print("Error loading src0")
#        exit(-1)
#    elif src[i] is None:
#        print("Error loading src",i)
#        exit(-1)
#    
#    # 合成
#    beta = (1.0 - alpha)
#    dst = cv2.addWeighted(src0, alpha, src[i], beta, 0.0)
#    cv2.imshow('dst',dst)
#    cv2.imwrite('img/add',i)
#    cv2.waitKey(0)
#
#cv2.destroyAllWindows()
#--------------------------------------------------------------------------