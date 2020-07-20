#OpenCVのインポート
import cv2
 
#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
cascade = cv2.CascadeClassifier("date/haarcascade_frontalface_default.xml")
 
#画像ファイルの読み込み
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow("frame",frame)

    key=cv2.waitKey(1)&0xFF

    if key==ord("q"):
        break

    if key==ord("s"):
        cv2.imwrite("photo.jpg",frame)

        img_gray = cv2.imread("photo.jpg")
        img_g=cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    
        #カスケード型分類器を使用して画像ファイルから顔部分を検出する
        face = cascade.detectMultiScale(img_g)
        
        #顔の座標を表示する
        print(face)
        
        #顔部分を切り取る
        for x,y,w,h in face:
            face_cut = img_g[y:y+h, x:x+w]
        
        # face_cut_color=cv2.cvtColor(face_cut,cv2.COLOR_GRAY2BGR)

        #画像の出力
        cv2.imwrite('face_cut.jpg', face_cut)
    
