
import cv2 
import numpy as np 
import argparse

def load_classes(path):
	with open(path,'r') as f:
		names = f.read().split('\n')
	return list(filter(None,names))			#filter removes empty string  

def run():
	net = cv2.dnn.readNet(opt.weights,opt.cfg)
	classes = load_classes(opt.names)

	layer_names = net.getLayerNames()
	outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors= np.random.uniform(0,255,size=(len(classes),3))

	#loading video from webcam 
	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX

	while True:
		_,frame = cap.read()
		height, width, channels = frame.shape

		blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce the frame to 320 * 320 pixels 

		net.setInput(blob)
		outs = net.forward(outputlayers)

		#Showing info on screen/ get confidence score of algorithm in detecting an object in blob
		class_ids=[]
		confidences=[]
		boxes=[]
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.3:
					#object detected
					center_x= int(detection[0]*width)
					center_y= int(detection[1]*height)
					w = int(detection[2]*width)
					h = int(detection[3]*height)

					#cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
					#rectangle co-ordinaters
					x=int(center_x - w/2)
					y=int(center_y - h/2)
					#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

					boxes.append([x,y,w,h]) #put all rectangle areas
					confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
					class_ids.append(class_id) #name of the object that was detected

		indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
		for i in range(len(boxes)):
			if i in indexes:
				x,y,w,h = boxes[i]
				if i==0:   
					label = str(classes[class_ids[i]])
					confidence= confidences[i]
					color = colors[class_ids[i]]
					cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
					cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,0.5,(255,255,255),1)

		cv2.imshow("Image",frame)
		key = cv2.waitKey(1)		#wait 1ms before the loop starts again and we process the next frame

		if key == 27:
			break;				#esc key stops the process 
	cap.release()
	cv2.destroyallwindows()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights',type = str,default = './yolov3-spp.weights',help = 'path to the weights file')
	parser.add_argument('--cfg', type = str, default = './yolov3-spp.cfg', help = 'path to the cfg file')
	parser.add_argument('--names', type = str, default = './coco.names', help = '*.names path')

	opt= parser.parse_args()
	print(opt)
	run()