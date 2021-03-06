import datetime
import base64
import json
import os
import io
import requests

import PIL
from PIL import ImageTk
from PIL import ImageGrab

import numpy as np
import cv2

import pyautogui
from tkinter import *
from threading import Thread

from google.cloud import vision

project_id = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'Google Cloud API Key'
client = vision.ImageAnnotatorClient()

URL = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=en"
API_KEY = "Azure Cloud API Key"
LOCATION = "southeastasia"

class Application():
	def __init__(self, master):
		self.master = master
		self.rect = None
		self.x = self.y = 0
		self.start_x = None
		self.start_y = None
		self.screenX1 = self.screenY1 = self.screenX2 = self.screenY2 = 0
		self.curX = None
		self.curY = None
		self.sr = True
		self.st = False
		# self.picture = np.array([])

		# root.configure(background = 'red')
		# root.attributes("-transparentcolor","red")

		root.attributes("-transparent", "blue")
		root.geometry('400x50+200+200')  # set new geometry
		root.title('Lil Snippy')
		self.menu_frame = Frame(master, bg="blue")
		self.menu_frame.pack(fill=BOTH, expand=YES)

		self.buttonBar = Frame(self.menu_frame,bg="")
		self.buttonBar.pack(fill=BOTH,expand=YES)

		self.snipButton = Button(self.buttonBar, width=3, command=self.createScreenCanvas, background="green")
		self.snipButton.pack(expand=YES)

		self.master_screen = Toplevel(root)
		self.master_screen.withdraw()
		self.master_screen.attributes("-transparent", "blue")
		#Set Resolution Frame
		self.picture_frame = Frame(self.master_screen, background = "blue")
		self.picture_frame.pack(fill=BOTH, expand=YES)

	def createScreenCanvas(self):
		self.master_screen.deiconify()
		root.withdraw()

		self.screenCanvas = Canvas(self.picture_frame, cursor="cross", bg="grey11")
		self.screenCanvas.pack(fill=BOTH, expand=YES)

		self.screenCanvas.bind("<ButtonPress-1>", self.on_button_press)
		self.screenCanvas.bind("<B1-Motion>", self.on_move_press)
		self.screenCanvas.bind("<ButtonRelease-1>", self.on_button_release)

		self.master_screen.attributes('-fullscreen', True)
		self.master_screen.attributes('-alpha', .3)
		self.master_screen.lift()
		self.master_screen.attributes("-topmost", True)

	def on_button_release(self, event):
		self.recPosition()

		if self.start_x <= self.curX and self.start_y <= self.curY:
			print(self.start_x, self.start_y, self.curX, self.curY)
			self.setWindowsSize(self.start_x, self.start_y, self.curX, self.curY)
			print("right down")

		elif self.start_x >= self.curX and self.start_y <= self.curY:
			print(self.curX, self.start_y, self.start_x, self.curY)
			self.setWindowsSize(self.curX, self.start_y, self.start_x, self.curY)
			print("left down")

		elif self.start_x <= self.curX and self.start_y >= self.curY:
			print(self.start_x, self.curY, self.curX, self.start_y)
			self.setWindowsSize(self.start_x, self.curY, self.curX, self.start_y)
			print("right up")

		elif self.start_x >= self.curX and self.start_y >= self.curY:
			print(self.curX, self.curY, self.start_x, self.start_y)
			self.setWindowsSize(self.curX, self.curY, self.start_x, self.start_y)
			print("left up")

		self.sr = True
		t1 = Thread(target=self.Recording_GUI).start()
		t2 = Thread(target=self.startRecording).start() 
		return event

	def setWindowsSize(self,x1,y1,x2,y2):
		self.screenX1 = x1
		self.screenY1 = y1
		self.screenX2 = x2
		self.screenY2 = y2

	def Recording_GUI(self):
		self.screenCanvas.destroy()
		self.master_screen.withdraw()

		self.sub_screen = Toplevel(self.master_screen)

		self.sub_screen.attributes("-transparent", "blue")
		self.sub_screen.geometry('400x50+200+200')

		self.button_sub = Frame(self.sub_screen,bg="")
		self.button_sub.pack(fill=BOTH,expand=YES)

		self.Button = Button(self.button_sub, width=3, command=self.exitScreenshotMode, background="red")
		self.Button.pack(expand=YES,side=LEFT)
		self.Button = Button(self.button_sub, width=3, command=self.takePhoto, background="green")
		self.Button.pack(expand=YES,side=RIGHT)

	def takePhoto(self):
		self.st = True

	def startRecording(self):
		while(self.sr == True):
			img = ImageGrab.grab(bbox=(self.screenX1,self.screenY1,self.screenX2,self.screenY2),all_screens=False) #bbox specifies specific region (bbox= x,y,width,height)
			img_np = np.array(img)
			frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
			
			#Image Processing
			if(self.st == True):
				self.st = False

				img_path = 'tmp/tmp.jpg'
				cv2.imwrite(img_path, frame) 
				print(img_path)
				
				file_name = os.path.abspath('tmp/tmp.jpg')
				with io.open(file_name, 'rb') as image_file:
					content = image_file.read()

				image = vision.Image(content=content)
				response = client.text_detection(image=image)
				text = response.text_annotations

				img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)

				print(text,"\n")

				for t in text:
					square = t.bounding_poly.vertices
					#print(square[0].x)
					character = t.description
					print(square,":",character)
					cv2.rectangle(img,(square[0].x,square[0].y),(square[1].x,square[2].y),(0,255,0),2)

				headers = {
					'Ocp-Apim-Subscription-Key': API_KEY,
					'Ocp-Apim-Subscription-Region': LOCATION,
					'Content-type': 'application/json'
				}

				data = [{"Text" : text[0].description}]

				request = requests.post(URL, headers=headers, json=data)
				response = request.json()

				print(text[0].description,"\n")
				print(request.text)

				cv2.imshow("Stack",img)
				
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

		print("Recording Stoped")

	def encode_image(self, image_path):
		with open(image_path, "rb") as f:
			image_content = f.read()
			return base64.b64encode(image_content)

	def exitScreenshotMode(self):
		print("Screenshot mode exited")
		self.screenCanvas.destroy()
		self.sub_screen.destroy()
		self.master_screen.withdraw()
		root.deiconify()
		self.sr = False

	def exit_application(self):
		print("Application exit")
		root.quit()

	def on_button_press(self, event):
		# save mouse drag start position
		self.start_x = self.screenCanvas.canvasx(event.x)
		self.start_y = self.screenCanvas.canvasy(event.y)

		self.rect = self.screenCanvas.create_rectangle(self.x, self.y, 1, 1, outline='red', width=3, fill="blue")

	def on_move_press(self, event):
		self.curX, self.curY = (event.x, event.y)
		# expand rectangle as you drag the mouse
		self.screenCanvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)

	def recPosition(self):
		print(self.start_x)
		print(self.start_y)
		print(self.curX)
		print(self.curY)

if __name__ == '__main__':
	root = Tk()
	app = Application(root)
	root.mainloop()

'''
while(True):
	img = ImageGrab.grab(bbox=(857,343,1840,853),all_screens=False) #bbox specifies specific region (bbox= x,y,width,height)
	img_np = np.array(img)
	frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
	cv2.imshow("test", img_np)
	cv2.waitKey(0)
cv2.destroyAllWindows()
'''