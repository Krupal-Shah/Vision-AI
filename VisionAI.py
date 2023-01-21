import numpy as np
from cv2 import cv2
from gtts import gTTS
import os
import time

net = cv2.dnn.readNet("yolov3.weights", "yolov_3.cfg")
classes = []
with open("coco1.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers_name = net.getLayerNames()
output_layers = [layers_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]


camera = cv2.VideoCapture(0)
def make_720p():
    camera.set(3, 1920)
    camera.set(4, 1080)

make_720p()

return_value, image = camera.read()

cv2.imwrite('opencv0.png', image)
time.sleep(0.5)

del(camera)

img = cv2.imread("opencv0.png")
img = cv2.resize(img, None, fx=0.85, fy=0.85)
height, width, channels = img.shape
lang = 'en'

blob = cv2.dnn.blobFromImage(img, 0.00392, (640, 480), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class_ids = []
confidences = []
boxes = []
Sum1 = ''
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        Sum1 = Sum1 + label + ','

output = gTTS(text=Sum1, lang=lang, slow=False)
output.save("Output.mp3")   
os.system("start Output.mp3")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

