
import cv2
import numpy as np


image = cv2.imread("img1.jpg")
cv2.imshow("image1", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image2 = cv2.imread("img1.jpg",0) #Tek bir kanal al.
cv2.imshow("img1-Renksiz", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Kanalsiz_image.jpg",image2 )

print(image)  #Her bir pikselin matris karşığıdır.
print(image2.size)
print(image2.shape)
print(image2.dtype)
print(image.shape)


#%%

image = cv2.imread("img2.jpg")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(image[(563,200)])
# Çıktı : [ 85  70 251]  563.satır 200. sütundaki pikselin BGR değerleri


print("Görüntünün Özellikleri : " + str(image.shape))
print("Görüntünün Veritipi : " + str(image.dtype))
print("Görüntünün Boyutu : " + str(image.size))


image[(50,30)] = [255,255,255] #50.satır 30. sütundaki piksel değeri değiştirildi.
cv2.imshow("Image-Degisme", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(100):
    image[70,i] = [0,0,0]
    
cv2.imshow("Image-Degisme", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

panda = cv2.imread("panda2.jpg")
panda[:,:,0] = 255 # 0 = Blue kanal 
cv2.imshow("Panda", panda)
cv2.waitKey(0)
cv2.destroyAllWindows()

panda[:,:,1] =255 #1 = Green Kanalı
cv2.imshow("Panda", panda)
cv2.waitKey(0)
cv2.destroyAllWindows()

panda[:,:,2] = 255  # 2 = Red Kanalı
cv2.imshow("Panda", panda)
cv2.waitKey(0)
cv2.destroyAllWindows()


panda = cv2.imread("panda2.jpg")
image[75:100,75:100,0] = 255
cv2.imshow("Panda", panda)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

mandalina = cv2.imread("mandalina.jpg")
cv2.imshow("Mandalina",mandalina)
cv2.waitKey(0)
cv2.destroyAllWindows()

kesit = mandalina[0:50,0:50]
cv2.imshow("ksit", kesit)

mandalina[50:100,100:150] = kesit
cv2.imshow("Kesit Yerleştirme",mandalina)
cv2.waitKey(0)
cv2.destroyAllWindows()

/#Dikkat : Kesit (50*50) 'likti bu yüzden kesiti image üzerine koyarken boyutuna uygun değerler seçilmelidir.#/

mandalina[50:100,100:150] = (0,0,0)
cv2.imshow("Kesit Boyama",mandalina)
cv2.waitKey(0)
cv2.destroyAllWindows()

#♦Seçilen noktayı (B,G,R) olarak istenilen dğerlerin verilmesi

#%%

adileNasit = cv2.imread("adileNasit.jpg")
cv2.imshow("Adile Nasit", adileNasit)
cv2.waitKey(0)
cv2.destroyAllWindows()


aynalama = cv2.copyMakeBorder(adileNasit,
                              75,75,125,125,cv2.BORDER_REFLECT)
cv2.imshow("Adile Nasit", aynalama)
cv2.waitKey(0)
cv2.destroyAllWindows()


uzatma = cv2.copyMakeBorder(adileNasit,120,120,120,120,cv2.BORDER_REPLICATE)
cv2.imshow("Adile Nasit", uzatma)
cv2.waitKey(0)
cv2.destroyAllWindows()




tekrar = cv2.copyMakeBorder(adileNasit,120,120,120,120,cv2.BORDER_WRAP)
cv2.imshow("Adile Nasit", tekrar)
cv2.waitKey(0)
cv2.destroyAllWindows()


sarma = cv2.copyMakeBorder(adileNasit,50,50,50,50,cv2.BORDER_CONSTANT,
                           value = (0,0,255))
cv2.imshow("Adile Nasit", sarma)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%%

hababam = cv2.imread("hababam.jpg")
cv2.rectangle(hababam,(360,220),(205,80),[0,0,255],9)
cv2.imshow("Hababam",hababam)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
(400,200) = rectangleın sol alt köşesi
(150,80) = sağ üst köşesi
[0,0,255] = rectangle rengi
9 = rectangle kalınlığı
*/
"""

#%% Ağırlıklı Toplama

emel = cv2.imread("img1.jpg")
türkan = cv2.imread("cicek.jpg")

print(emel[100,200])
print(türkan[200,40])
print(emel[100,200] + türkan[300,430])

#%%% Ağırıklı Toplama

image1 = cv2.imread("img1.jpg")
image2 = cv2.imread("cicek.jpg")

toplam = cv2.add(image1,image2)
cv2.imshow("toplam image", toplam)
cv2.waitKey(0)
cv2.destroyAllWindows()


agırlıkToplama = cv2.addWeighted(image1,0.7,image2,0.3,0)
cv2.imshow("agırlık image", agırlıkToplama)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%%
cicek = cv2.imread("img1.jpg")
gricicek = cv2.cvtColor(cicek, cv2.COLOR_BGR2GRAY)

cv2.imshow("orjinal çicek", cicek)
cv2.imshow("gri cicek", gricicek)
cv2.waitKey(0)
cv2.destroyAllWindows()

satir,sütun,channel = cicek.shape
print("Orjinal ", satir, sütun,channel)

satir2,sütun2 = gricicek.shape
print("gri image", satir2,sütun2)

#%%

hababam= cv2.imread("hababam.jpg")
ikikat = cv2.pyrUp(hababam)
ikikatkucuk = cv2.pyrDown(hababam)

cv2.imshow("hababam", hababam)
cv2.imshow("Iki kat", ikikat)
cv2.imshow("Iki kat kucuk ", ikikatkucuk)
cv2.waitKey(0)
cv2.destroyAllWindows()

#pyrUp () = image'in geişliğinden iki kat boyundan iki kat  artırdı.
#pyrDown ()= image'in genişliğinden iki kat boyundan iki kat  azalttı.

satir,sütun,kanal = hababam.shape
ikikatsatir,ikikatütun,ikikatkanal = ikikat.shape
satirkucuk,sütunkucuk,kanal= ikikatkucuk.shape

print("orjinal", satir,sütun,kanal)
print("orjinal",ikikatsatir,ikikatütun,ikikatkanal)
print("orjinal",satirkucuk,sütunkucuk)


#%%

image = np.zeros((300,300,3), dtype = "uint8")

print(image)

#%% Kameradan Canlı Görüntü Alma

kamera = cv2.VideoCapture(0)

while True:
    ret,goruntu = kamera.read()
    
    cv2.imshow("Me", goruntu)
    
    if cv2.waitKey(30)  & 0xFF == ('q'):
        break
    
kamera.release()
cv2.destroyAllWindows()
    
#%%

image =np.zeros((300,300,3), dtype ="uint8")
cv2.line(image, (0,0), (100,100), (0,0,255),3)
cv2.circle(image,(150,150), 25,(0,255,0),3)
cv2.putText(image, "puttext", (100,200), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
cv2.imshow("Cizimler", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% 

kamera = cv2.VideoCapture(0)

while True:
    ret,goruntu = kamera.read()
    cv2.rectangle(goruntu,(100,100), (200,200), [0,0,255],3)
    cv2.line(goruntu, (0,0),(50,50),(255,0,0),3)
    cv2.circle(goruntu,(250,250), 25,(0,255,0),3)
    cv2.putText(goruntu, "Deneme", (220,220), cv2.FONT_HERSHEY_DUPLEX,1,(250,250,250),1)
    cv2.imshow("Deneme", goruntu)
    
    if cv2.waitKey(20)  & 0xFF == ('q'):
        break
    
kamera.release()
cv2.destroyAllWindows()

#%%%

import cv2
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("test_ video.mp4")
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#%%

import numpy as np 

img = cv2.imread('lena.png')
imgHor = np.hstack((img,img))
imgVer = np.vstack((img,img))

cv2.imshow("Horizontal",imgHor)
cv2.imshow("Vertical",imgVer)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
İki görüntünün aynı kana sayısına sahip olması gerekir. Aksi takdir de birleştirmeler gerçekleştirilemez.
"""































