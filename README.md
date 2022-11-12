<!--
# ImageProcessingOpencvPython
Image Processing using OPENCV and python
Title:Newspaper Headlines Extraction
This is an image analysis system that can detect and extract newspaper headlines from given newspaper iamges using Opencv and python. In this python project,
we’re going to make a text detector and extractor from an image 
using opencv and OCR. 
We’ll use the Tesseract engine to perform the character recognition system and the pytesseract python package to interact with Tesseract in python.
Firstly, read the image in color mode. We have to find contours before find contours image must be in binary form.
Apply RGB to binary conversion first of all convert RGB image to grayscale image 
and then apply thresholding method using combine with binary thresh and 
thresh _Otsu. 
Then find contours apply contours height heuristic. Create blank image of the same dimension of the original image. Find average height from all the contours 
heights and Draw those which are above the average. We got the title contours.
Apply RLSA algorithm to connect the title contours and avoid noise with some heuristics. 
Apply width heuristic for filter contours based on width. Finally, pass the images to pytesseract to get the text
Objectives:In this OPENCV python project, we’ve built a text detector and extractor. This project describes an image analysis system that extracts newspaper headlines.
-->
