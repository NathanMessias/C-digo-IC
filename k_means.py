import numpy as np
import cv2

img = cv2.imread('imgred03l.jpg')
# img = cv2.medianBlur(img, 3)
img2 = img
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
K = 2
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('Original', img2)
cv2.imshow('K-means', res2)

# cv2.imwrite("k-means3.jpg", res2)

cv2.waitKey(0)
cv2.destroyAllWindows()
