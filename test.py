import cv2
import numpy as np 
import matplotlib.pyplot as plt
#import tensorflow as tf

WI = 1280
HI = 720

def stackImages(scale,imgArray):
	rows = len(imgArray)
	cols = len(imgArray[0])
	rowsAvailable = isinstance(imgArray[0], list)
	width = imgArray[0][0].shape[1]
	height = imgArray[0][0].shape[0]
	if rowsAvailable:
		for x in range ( 0, rows):
			for y in range(0, cols):
				if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
					imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
				else:
					imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
				if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
		imageBlank = np.zeros((height, width, 3), np.uint8)
		hor = [imageBlank]*rows
		hor_con = [imageBlank]*rows
		for x in range(0, rows):
			hor[x] = np.hstack(imgArray[x])
		ver = np.vstack(hor)
	else:
		for x in range(0, rows):
			if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
				imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
			else:
				imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
			if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
		hor= np.hstack(imgArray)
		ver = hor
	return ver

def getCont(img):
	cont,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	maxArea = 0
	biggest = np.array([])
	for c in cont:
		area = cv2.contourArea(c)
		#cv2.drawContours(img_copy,c,-1,(255,0,0),3)
		if area >= 5000:
			peri = cv2.arcLength(c,True)
			approx = cv2.approxPolyDP(c,0.02*peri,True)
			if area > maxArea and len(approx) == 4:
				biggest = approx
				maxArea = area
	cv2.drawContours(img_copy,biggest,-1,(255,0,0),20)	
	return biggest
		#x,y,w,h = cv2.boundingRect(approx)
		#cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)

def reorder(biggest):
	#Reshape Numpy Array
	myPoints = biggest.reshape((4,2))
	myPointsNew = np.zeros((4,1,2),np.int32)
	
	sum = myPoints.sum(1)
	myPointsNew[0] = myPoints[np.argmin(sum)]
	myPointsNew[3] = myPoints[np.argmax(sum)] 
	
	diff = np.diff(myPoints,axis = 1)
	myPointsNew[1] = myPoints[np.argmin(diff)]
	myPointsNew[2] = myPoints[np.argmax(diff)] 

	return myPointsNew

def getWrap(img,biggest):
	biggest = reorder(biggest)
	pts1 = np.float32(biggest)
	pts2 = np.float32([[0, 0], [WI, 0], [0, HI], [WI, HI]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	imgOutput = cv2.warpPerspective(img, matrix, (WI, HI))

	imgOutput = imgOutput[10:imgOutput.shape[0]-10,10:imgOutput.shape[1]-10]
	imgOutput = cv2.resize(imgOutput,(WI,HI))

	return imgOutput

def preProccessing(img):
	Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	Blur = cv2.GaussianBlur(Gray,(5,5),1)
	Canny = cv2.Canny(Blur,130,130)
	#Thicken The Image
	kernel = np.ones((5,5))
	Dial = cv2.dilate(Canny,kernel,iterations=2)
	Thres = cv2.erode(Dial,kernel,iterations=1)
	return Canny,Thres

def squareCorner(img):
	Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	Canny = cv2.Canny(Gray,80,80)
	#Thicken The Image
	kernel = np.ones((5,5))
	Dial = cv2.dilate(Canny,kernel,iterations=2)
	Thres = cv2.erode(Dial,kernel,iterations=1)
	cont,hier = cv2.findContours(Thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	maxArea = 0
	square = []
	for c in cont:
		area = cv2.contourArea(c)
		if area <= 2000:
			peri = cv2.arcLength(c,True)
			approx = cv2.approxPolyDP(c,0.2*peri,True)
			x,y,w,h = cv2.boundingRect(approx)
			square.append([x-20,y,w+40,h])
	return square,Dial,Thres
'''
cap = cv2.VideoCapture(0)
cap.set(10, 150)
while True:
	status, img = cap.read()
	img = cv2.resize(img,(WI,HI))
	img_copy = img.copy()
	Dial,Thres = preProccessing(img)
	biggest = getCont(Thres)
	img_warped = np.zeros_like(img)
	if biggest.shape[0] != 0:
		img_warped = getWrap(img_copy,biggest)
	Stack = stackImages(0.4,([img,img_copy],[img_warped,Thres]))
	cv2.imshow("test",Stack)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()#Release Webcam use


#out.release()
cv2.destroyAllWindows()

cv2.rectangle(img_warped,(x,y),(x+w,y+h),(0,255,0),2)
'''
#OD = tf.keras.models.load_model('C:/Users/ASUS/Documents/Program/Python/AI/Computer Vision/Character Recognition/ocr_test2')
#OD.summary()
img = cv2.imread("C:/Users/ASUS/Downloads/Image/test3.jpg",cv2.IMREAD_UNCHANGED)
#img = cv2.resize(img, (WI,HI), interpolation=cv2.INTER_LINEAR)
img_copy = img.copy()
Dial,Thres = preProccessing(img)
biggest = getCont(Thres)
#img_warped = img.copy()
img_warped = np.zeros_like(img)
if biggest.shape[0] != 0:
	img_warped = getWrap(img_copy,biggest)
square,Dial2,Thres2 = squareCorner(img_warped)
'''
img_squared = img_warped.copy()
labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
text = []
for i in square:
	cv2.rectangle(img_squared,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),2)
	crop_img = img_warped[i[1]:i[1]+i[3], i[0]:i[0]+i[2]]
	#cv2.imshow("Stack",crop_img)
	#cv2.waitKey(0)
	print(crop_img.size)
	if crop_img.size != 0:
		crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
		crop_img = cv2.resize(crop_img, (28,28), interpolation=cv2.INTER_LINEAR)
		crop_img = (255-crop_img)
		#cv2.imshow("Stack",crop_img)
		#cv2.waitKey(0)
		crop_img = crop_img[..., np.newaxis]
		crop_img = crop_img.astype('float32')
		crop_img = crop_img/255.0
		crop_img = np.array([crop_img])
		print(crop_img.shape)
		prediction = OD.predict(crop_img)
		top_k_values, top_k_indices = tf.nn.top_k(prediction)
		print(top_k_indices," ",top_k_values)
		an_array = top_k_indices.numpy()
		cv2.putText(img_squared,labels[an_array[0][0]],
		(i[0]+(i[2]//2)-20,i[1]+(i[3]//2)-50),cv2.FONT_HERSHEY_COMPLEX,0.7,
		(255,0,0),2)
		text.append(labels[an_array[0][0]])
		#plt.barh(labels,prediction[0].tolist()) 
		#plt.show()
	
print(f"\n{text}")
'''
#img_copy = cv2.resize(img_copy, (1000,1000), interpolation=cv2.INTER_LINEAR)
imgBlank = np.zeros_like(img)
Stack = stackImages(0.7,([img,Thres,img_copy],[img_warped,Thres2,imgBlank]))
cv2.imshow("Stack",Stack)
#cv2.imshow("test",Thres2)
cv2.waitKey(0)
cv2.destroyAllWindows()
