import cv2 as cv
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

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

def main():
    if len(sys.argv) < 3:
        print("Error: Need to have image list and the region side count in command!")
        exit(1)

    print("Loading images...")
    fileName = sys.argv[1]
    regionSideCount = sys.argv[2]

    allImages = cv.imread([f for f in listdir(fileName) if isfile(join(fileName, f))])

    featureSize = getFeatureLength(regionSideCount)
    feature = np.zeros(1, featureSize, cv.CV_64FC1)

    for f in allImages:
        logf = log(f)
        cv.imshow("Laplacian of Gaussian", logf)
        hogf = computeFeatures(f, regionSideCount, )
        cv.imshow("Poor Man's HOG", hogf)

if __name__ == "__main__":
    main()