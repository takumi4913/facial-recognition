#OpenCVのインポート
import cv2
from PIL import Image
 
#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
cascade = cv2.CascadeClassifier("date/haarcascade_frontalface_default.xml")
 
#画像ファイルの読み込み
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow("x de syu-ryou",frame)

    key=cv2.waitKey(1)&0xFF

    if key==ord("q"):
        break

    if key==ord("s"):
        cv2.imwrite("photo.png",frame)

        img_gray = cv2.imread("photo.png")
        img_g=cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    
        #カスケード型分類器を使用して画像ファイルから顔部分を検出する
        face = cascade.detectMultiScale(img_g)
        
        #顔の座標を表示する
        print(face)
        
        #顔部分を切り取る
        for x,y,w,h in face:
            face_cut = img_gray[y:y+h, x:x+w]

        #画像の出力
        cv2.imwrite('face_cut_user.png', face_cut)

img = Image.open('face_cut_user.png')
img_resize = img.resize((225, 225))
img_resize.save('user.png')
# 画像取り込み

src0 = cv2.imread(cv2.samples.findFile('user.png'))
src = [cv2.imread(cv2.samples.findFile('addweighted/0.png')),
        cv2.imread(cv2.samples.findFile('addweighted/1.png')),
        cv2.imread(cv2.samples.findFile('addweighted/2.png')),
        cv2.imread(cv2.samples.findFile('addweighted/3.png')), 
        cv2.imread(cv2.samples.findFile('addweighted/5.png')),
        cv2.imread(cv2.samples.findFile('addweighted/6.png'))]

# 特定の表情色・imageと合成
for i in range(6):
    if src0 is None:
        print("Error loading src0")
        exit(-1)
    elif src[i] is None:
        print("Error loading src",i)
        exit(-1)
    
    # 合成
    dst = cv2.addWeighted(src0, 0.5, src[i], 0.5, 0.0)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)

cv2.destroyAllWindows()
