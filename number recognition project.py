
import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
image = cv2.imread('licensed_car10.jpeg')
image = imutils.resize(image, width=500)
cv2.imshow("Original image_brat", image)   #to display original image
cv2.waitKey(0)

#gray scaling the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale image_brat", gray)
cv2.waitKey(0)

#Removing the noise(smoothing)
gray = cv2.bilateralFilter(gray, 11, 19, 19)
cv2.imshow("Smoother image_brat", gray)
cv2.waitKey(0)

#thresholding the image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canned image_brat", edged)
cv2.waitKey(0)

#countouring the image
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Canned image after contour_brat", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCount = None   #we currently have no Number plate contour

count = 0
name = 1

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
    if(len(approx)==4):
        NumberPlateCount= approx
        x, y, w, h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        cv2.imwrite(str(name) + '.png', crp_img)
        name +=1
        break

#number plate detected and extracted
cv2.drawContours(image,[NumberPlateCount], -1, (0, 255, 0), 3)
cv2.imshow("Final image_brat", image)
cv2.waitKey(0)

#save the cropped image
crop_img_loc = '1.png'
cv2.imshow("Cropped Image_brat", cv2.imread(crop_img_loc))


#to display the text recognised
text = pytesseract.image_to_string(crop_img_loc, lang='eng')
print("Number is: ",text)
cv2.waitKey(0)







