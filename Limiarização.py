import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Lendo a Imagem em escala de cinza
img = cv2.imread('imgred03l.jpg',0)
# Realização dos testes com as funções thresh
ret,thresh1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,130,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,150,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,150,255,cv2.THRESH_TOZERO_INV)
# Utilizando a Biblioteca Matplotlib para exibir os resultados
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
# Salvando os contornos de acordo com o resultado da Limiarização: BINARY_INV
contours, hierarchy = cv.findContours(thresh2,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
# Mostrando a quantidade de contornos salva
print(contours)
print('Número de contornos na imagem=' + str(len(contours)))
# Desenhando os contornos na imagem original
cv.drawContours(img,contours,-1,(0,255,0),3)

# Visualizando os resultados do desenho na imagem
cv.imshow("Original", img)
cv.imshow("desenho", img)

# Salvando os resultados
#cv.imwrite("imgred03l1.jpg", thresh1)

cv.waitKey()
cv.destroyAllWindows()