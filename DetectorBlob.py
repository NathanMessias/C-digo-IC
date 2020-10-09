# Standard imports
import cv2
import numpy as np

# Read image
im = cv2.imread("imgred02t.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original', im)
print(im.shape)
im = cv2.resize(im, (580, 584))  # 806 604    /505 497

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 1  # 10
params.maxThreshold = 200  # 200
params.thresholdStep = 1

# Filter by Area.
params.filterByArea = True  # True
params.minArea = 30  # 1500

# Filter by Circularity
params.filterByCircularity = True  # True
params.minCircularity = 0.785  # 0.1

# Filter by Convexity
params.filterByConvexity = True  # True
params.minConvexity = 0.7  # 0.87  0.82

# Filter by Inertia
#params.filterByInertia = True  # True
#params.minInertiaRatio = 0.1  # 0.01

# Filter by
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
total_count = 0
for i in keypoints:
    total_count = total_count + 1

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
print(total_count)
cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)

