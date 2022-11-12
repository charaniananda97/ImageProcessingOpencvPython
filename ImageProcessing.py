import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import math
from pythonRLSA import rlsa
from PIL import Image

#1)Reading the image
img = cv2.imread(r'E:\314 project\ImageProcessingOpencvPython\sample images\SLIIT-Business-School.jpg'
                 ,cv2.IMREAD_COLOR)
img_ = img.copy()

img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilation_1 = cv2.dilate(img1,kernel_1,iterations=1)
dilation_1 = cv2.cvtColor(dilation_1,cv2.COLOR_BGR2RGB)
#cv2.imshow('dilation', dilation_1)

#Convert to gray image
gray_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 

#convert grayscale image to binary image
ret,thresh1 = cv2.threshold(gray_img1,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('Binary Image', thresh1)


#2)Detect edges
edges = cv2.Canny(thresh1,100,200)
#cv2.imshow('edges', edges)

#create blank image of same dimension of the original image
mask = np.ones(img1.shape[:2], dtype="uint8") * 255
contours,heuristic = cv2.findContours(~thresh1,cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
#collecting heights of each contour
heights = [cv2.boundingRect(contour)[3] for contour in contours] 
avgheight = sum(heights)/len(heights) # average height

thresh1_copy = thresh1.copy()
thresh1_copy = cv2.cvtColor(thresh1_copy,cv2.COLOR_GRAY2BGR)

contours,heuristic = cv2.findContours(~thresh1,cv2.RETR_EXTERNAL,

                                      cv2.CHAIN_APPROX_SIMPLE)
#3)find contours
for contour in contours:
    """draw a rectangle around those contours on main image
    """
    [x,y,w,h] = cv2.boundingRect(contour)
    cv2.rectangle(thresh1_copy, (x,y), (x+w,y+h), (0, 255, 0), 1)

#cv2.imshow('Contours', thresh1_copy)


#4)inner contours of the second largest contour    
second_largest_cont = sorted(contours, key = cv2.contourArea, reverse = True)[1:2]


#5)finding the larger contours
#Applying Height heuristic
for second_largest_cont in contours:
    [x,y,w,h] = cv2.boundingRect(second_largest_cont)
    if h > 2*avgheight:
        cv2.drawContours(mask, [second_largest_cont], -1, (0,255,0), 3)
#cv2.imshow('mask',mask)
#cv2.imwrite('filter.png', mask)


#6)
x, y = mask.shape # image dimensions

value = max(math.ceil(x/100),math.ceil(y/100))+20
mask = rlsa.rlsa(mask, True, False, value) #rlsa application
#cv2.imshow('mask1', mask)

contours,heuristic = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


#7)
mask2 = np.ones(img1.shape, dtype="uint8") * 255 # blank 3 layer image
for second_largest_cont in contours:
    [x, y, w, h] = cv2.boundingRect(second_largest_cont)
    if w > 0.60*img1.shape[1]:# width heuristic applied
        title = img1[y: y+h, x: x+w] 
        mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
        img1[y: y+h, x: x+w] = 255 # nullified the title contour on original image

mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2RGB)
#cv2.imshow('Mask2', mask2)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)


#8)
title = pytesseract.image_to_string(Image.fromarray(mask2))
content = pytesseract.image_to_string(Image.fromarray(img1))
print('title - {0}'.format(title))


#x = cv2.resize(mask2,dsize=(400,400))
mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2RGB)

#img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow('Original image',img)
cv2.imshow('title',mask2)
cv2.imwrite('title.png', mask2)
#cv2.imshow('content', img1)

plt.style.use('grayscale')
plt.figure().patch.set_facecolor('white')
plt.suptitle("Headlines Extraction",fontsize=10,color='blue')

plt.subplot(231)
plt.title("Original image",fontsize = 8)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(img1)

plt.subplot(232)
plt.title("Gray image",fontsize = 8)
plt.axis('off')
plt.imshow(gray_img1)

plt.subplot(233)
plt.title("Binary image",fontsize = 8)
plt.axis('off')
plt.imshow(thresh1)

plt.subplot(234)
plt.title("contours_image",fontsize = 8)
plt.axis('off')
plt.imshow(thresh1_copy)

plt.subplot(235)
plt.title("Mask",fontsize = 8)
plt.axis('off')
plt.imshow(mask)

plt.subplot(236)
plt.title("Mask2",fontsize = 8)
plt.axis('off')
plt.imshow(mask2)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()
