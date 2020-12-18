import cv2
import matplotlib.pyplot as plt

import numpy as np

img = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE) 
#cv2.imshow("img1",img)
#cv2.IMREAD_COLOR

plt.imshow(img,  cmap='gray', interpolation = 'bicubic')
plt.plot([50,100],[100,100], 'c', linewidth =5)
plt.show()
#%%

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
camera = cv2.VideoCapture(0)
while True :
    ret, goruntu = camera.read()
    gray = cv2.cvtColor (goruntu, cv2.COLOR_BGR2GRAY)
    out.write(goruntu)
    cv2.imshow("frame",goruntu)
    cv2.imshow("frame", gray)
    
    
    if cv2.waitKey(1) & 0XFF == ord ('q'):
        break
        
camera.release()
out.release()
cv2.destroyAllWindows()
 
#%%

plt = np.array([[10,5],[20,30],[70,50],[50,10]])
cv2.polylines(img, [plt], True, (0,0,0),3)
cv2.imshow("lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
img = cv2.imread("img1.jpg")
img [100:250,300:400]  = [0,255,0]
cv2.imshow("kesit", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Thresholding 

mum = cv2.imread("mum.jpg")

retral, threshold = cv2.threshold(mum, 12,255, cv2.THRESH_BINARY )

cv2.imshow("threshld", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

grayscaled = cv2.cvtColor (mum, cv2.COLOR_BGR2GRAY)

retral2, threshold2 = cv2.threshold(grayscaled, 12,255, cv2.THRESH_BINARY )
cv2.imshow("threshld", threshold2)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Color Filtering

camera =cv2.VideoCapture(0)

while True:
    _,goruntu = camera.read()
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    
    lower = np.array([150,150,150])
    
    upper = np.array([150,150,150])

    mask = cv2.inRange(hsv,lower,upper)
    res = cv2.bitwise_and(goruntu,goruntu, mask=mask)



   #%% Smoothing and Blurring 
    kernel = np.ones((15,15), np.float32) /255
    
    smoothed = cv2.filter2D(img, -1, kernel)
    
    bulur = cv2.GaussianBlur(img, (15,15), 0)
    
    median = cv2.meadianBulur(img,15)
    
    #%% Morphological Transformations 
    
    kernel = np.ones((5,5), uint8)
    erosion = cv2.erode (mask,kernel,iterations = 1)
    dilatin = cv2.dilate (mask,kernel,iterations = 1)
    
    opening =  cv2.MorphologyEX(mask, cv2.MORPH_OPEN, kernel)
    closing  =  cv2.MorphologyEX(mask, cv2.MORPH_CLOSE, kernel)
    
    
#%% Edge Detection

laplacian = cv2.Laplacian(frame,  cv2.CV_64F)
sobelx = cv2.Sobel(frame,cv2.CV_64F, 1, 0, ksiz =5 )
sobelx = cv2.Sobel(frame,cv2.CV_64F, 0, 1, ksiz =5 )
edges = cv2.Canny(frame, 100,200)


    
#%%
    
    cv2.imshow("mask", mask)
    cv2.imshow("smoothed", smoothed)
    cv2.imshow("bulur", bulur)
    cv2.imshow("median", median)
    
    k = cv2.waitKey(5) & 0xFF 
        if k == 27 : 
    
            break
        
camera.release()
cv2.destroyAllWindows()

#%%% Template Matching


img_rgb = cv2.imread('img3.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('template.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)

#%% Corner Detection

img = cv2.imread('corner.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow('Corner',img)

#%% Feature Matching


img1 = cv2.imread('feature.jpg',0)
img2 = cv2.imread('feature2.jpg',0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)



img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()

#%% MOG BackGround Reduction

cap = cv2.VideoCapture('people.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()




