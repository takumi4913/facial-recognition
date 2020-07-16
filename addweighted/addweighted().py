import cv2
alpha = 0.5

# 画像取り込み
src0 = cv2.imread(cv2.samples.findFile('user.png'))
src = [cv2.imread(cv2.samples.findFile('0.png')),
        cv2.imread(cv2.samples.findFile('1.png')),
        cv2.imread(cv2.samples.findFile('2.png')),
        cv2.imread(cv2.samples.findFile('3.png')),
        cv2.imread(cv2.samples.findFile('5.png')),
        cv2.imread(cv2.samples.findFile('6.png')),]

# α値を入力
print('''0.3~0.8の間から入力
-----------------------
* Enter alpha [0.0-1.0]: ''')
input_alpha = float(input().strip())
if 0 <= alpha <= 1:
    alpha = input_alpha

# 特定の表情色・imageと合成
for i in range(6):
    if src0 is None:
        print("Error loading src0")
        exit(-1)
    elif src[i] is None:
        print("Error loading src",i)
        exit(-1)
    
    # 合成
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(src0, alpha, src[i], beta, 0.0)
    cv2.imshow('dst',dst)
    cv2.imwrite('img/add',i)
    cv2.waitKey(0)

cv2.destroyAllWindows()
