import numpy as np
import cv2

image = cv2.imread('./img/1.jpg')
#image = cv2.resize(image, (32, 32))

image = image[0:400,250:550]

cv2.imwrite('./1_re.jpg', image)
