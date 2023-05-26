import cv2
import os
import urllib
import urllib.request


recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
names = []

def face_detect_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier('D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
        cv2.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=1)
        ids, confidence = recogizer.predict(gray[y:y+h, x:x+w])
        if confidence < 60:
            cv2.putText(img, str(names[ids-1]) + ' ' + str(100 - confidence), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result', img)

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 720
    height_new = 540
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def name():
    path = './data'
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.', 2)[1])
       if not names.__contains__(name):
           names.append(name)


name()
# print(names)
img = cv2.imread('face10.png')
img = img_resize(img)
face_detect_demo(img)
while True:
    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()
