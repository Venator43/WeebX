from PIL import ImageGrab
import numpy as np
import cv2

img = cv2.imread("C:/Users/ASUS/Downloads/Image/test3.jpg",cv2.IMREAD_UNCHANGED)
img2 = cv2.imread("C:/Users/ASUS/Downloads/Image/test3.jpg",cv2.IMREAD_UNCHANGED)

'''
while(True):
	img = ImageGrab.grab(all_screens=True) #bbox specifies specific region (bbox= x,y,width,height)
	img_np = np.array(img)
	frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
	cv2.imshow("test", frame)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
'''