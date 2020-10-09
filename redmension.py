# Importação da Biblioteca OpenCV
import cv2 as cv

img = cv.imread('img02.jpg', 0)

img = cv.medianBlur(img,7)

scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

 #resize image
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

#crop image       Altura x  Largura
cropped = resized[167:664, 60:554]

clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
cl1 = clahe.apply(cropped)

#cv.imshow('clahe_2.jpg',cl1)

print('Resized Dimensions : ', resized.shape)
cv.imshow('Redimensionado', resized)
cv.imshow('Recortado', cropped)
cv.imshow('Equalizador', cl1)
#cv.imshow("Recortada", cropped)

#cv.imwrite("imgred02l.jpg", cropped)

cv.waitKey(0)
cv.destroyAllWindows()