import cv2 as cv
import medpy
import png
import SimpleITK as sitk
import numpy as np
import scipy.misc
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join

#File Heirarchy 
#LUNA16DATA
#|
#|subsets
#|       |
#|       |subset#
#|
#|Output
#        |
#        |subset#

def log(image):
    gauss = cv.GaussianBlur(image, (3, 3), 0)
    laplacian = cv.Laplacian(gauss, cv.CV_32F, 3, 0.25)
    return laplacian

def computeOneFeatures(image, feature, startIndex):
    magImg = np.zeros((256, 256, 1), dtype="float64")
    angleImg = np.zeros((256, 256, 1), dtype="float64")
    gauss = cv.GaussianBlur(image, (3, 3), 0)
    sobelX = cv.Sobel(gauss, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(gauss, cv.CV_64F, 0, 1)
    sumgrad = 0.0

    cv.cartToPolar(sobelX, sobelY, magImg, angleInDegrees=True)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            index = np.floor((angleImg[x, y]) / 10)
            feature[0, startIndex + index] += magImg[x, y]
            sumgrad += magImg[x, y]

    if sumgrad > 0:
        for i in range(36):
            feature[0, startIndex + i] /= sumgrad

def getFeatureLength(regionSideCount):
    return (regionSideCount * regionSideCount) * 36

def computeFeatures(image, regionSideCount, feature):
    regionSize = image[1]/regionSideCount
    index = 0

    for i in range (feature[1]):
        feature[0, i] = 0

    for i in i < regionSideCount:
        for j in j < regionSideCount:
            subImg = image(cv.rectangle(j * regionSize), (i * regionSize), regionSize, regionSize)
            computeOneFeatures(subImg, feature, index)
            index += 36

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    return numpyImage            


baseSubsetDirectory = r'LUNA16Data/subsets/subset'
targetSubsetDirBase = r'LUNA16Data/Output/subset'
NUM_SUBSET = 2
regionSideCount = 10
    
    
#read and convert to pngs, output is png folder    
for setNumber in range(NUM_SUBSET):
    subsetDirectory = baseSubsetDirectory + str(setNumber)
    list = listdir(subsetDirectory)
    subsetList = []
        
for file in list:
    if file.endswith(".mhd"):
            subsetList.append(file)
                
for file in subsetList:
    try:
        fileName = file[:-4]
        filePath = subsetDirectory + '/' + file
        try:
            numpyimage = load_itk_image(filePath)
            #working np.save(targetSubsetDirBase + str(setNumber) + '/' + fileName + 'n', numpyimage)
            #image = rawData_to_png(filePath)
            
            #1kb pngs not right png.from_array(numpyimage, mode='L').save(targetSubsetDirBase + str(setNumber) + '/' + fileName + 'p' + '.png')
            
        
            
            logFile = open('log_what.txt', 'a')
            logFile.write(str(setNumber) + ' -> ' + fileName + '\n')
            logFile.close()
        except:
            logFile = open('log_what.txt', 'a')
            logFile.write('ERROR: '+ str(setNumber) + ' -> ' + fileName + '\n')
            logFile.close()
            
            
#log all pngs -log(png)
#hog all pngs -computefeatures(png)
#pass log and hog to model
#trainlog/validatelog/testlog
#trainhog/validatehog/testhog

        featureSize = getFeatureLength(regionSideCount)
        feature = np.zeros(1, featureSize, cv.CV_64FC1)
    
        #logf = log(image)
        #cv.imshow("Laplacian of Gaussian", logf)
        #hogf = computeFeatures(image, regionSideCount, )
        #cv.imshow("Poor Man's HOG", hogf)
        
#write pngs to png folder
#write hogs to hog folder
#write logs to log folder
                
        
        
        #cv.imwrite(targetSubsetDirBase + str(setNumber) + '/' + fileName + 'l', logf)
        #cv.imwrite(targetSubsetDirBase + str(setNumber) + '/' + fileName + 'h', hogf)
        
        
        logFile = open('log_processing.txt', 'a')
        logFile.write(str(setNumber) + ' -> ' + fileName + '\n')
        logFile.close()
    except:
        logFile = open('log_processing.txt', 'a')
        logFile.write('ERROR: '+ str(setNumber) + ' -> ' + fileName + '\n')
        logFile.close()
               