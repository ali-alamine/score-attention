import pickle
import numpy as np
import matplotlib.pyplot as plt

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
from gaze_tracking import GazeTracking


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Analyze!")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)
        

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)

        p = []
        ll = []
        lr = []
        lc = []
        for key,value in results.items():
            p.append("Person # "+str(key))
            lc.append(int(value[0]))
            ll.append(int(value[1]))
            lr.append(int(value[2]))
        
        X_axis = np.arange(len(p))

        plt.bar(X_axis +0.20, ll, width=0.2, label = 'Looking Left')
        plt.bar(X_axis +0.20*2, lr, width=0.2, label = 'Looking Right')
        plt.bar(X_axis +0.20*3, lc, width=0.2, label = 'Looking Center')

        plt.xticks(X_axis, p)
        plt.legend()

        plt.xlabel("Persons")
        plt.ylabel("Number of Frames")
        plt.title("Analytics Plot")
        plt.show()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def run(self):

        # loading gaze tracking object
        gaze = GazeTracking()

        # loading SSD and ResNet network based caffe model for 300x300 dim imgs
        net = cv2.dnn.readNetFromCaffe("weights-prototxt.txt", "res_ssd_300Dim.caffeModel")

        # variable containing frame number
        frame_number = 0

        # dict containing attention of each student
        self.person= {
            0:{
                0:0,
                1:0,
                2:0
            },
            1:{
                0:0,
                1:0,
                2:0
            },
            2:{
                0:0,
                1:0,
                2:0
            }
        }

        detected_frames = []

        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # incrementing frame count
                frame_number += 1

                # # Convert into grayscale
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # resizing image to 300x300
                (height, width) = frame.shape[:2]
                
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))

                # pass the blob into the network
                net.setInput(blob)

                # detected faces
                detections = net.forward()

                if detections.shape[2]>0:
                    detected_frames.append(0)


                # loop over the detections to extract specific confidence
                for i in range(0, detections.shape[2]):
                    self.text = ""

                    # extract the confidence (i.e., probability) associated with the prediction
                    confidence = detections[0, 0, i, 2]

                    # greater than the minimum confidence
                    if confidence > 0.5:

                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (x1, y1, x2, y2) = box.astype("int")
                
                        # draw the bounding box of the face along with the associated probability
                        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 0, 255), 2)

                        # cropping face from frame image
                        crop_img = frame[y1:y2, x1:x2]

                        try: 
                            # predicting gaze for a face
                            gaze.refresh(crop_img)  
                        except:
                            continue
                        
                        # new_frame = gaze.annotated_frame()

                        if gaze.is_right():
                            self.person[i][2]+=1
                            self.text = f"Person #{i} Looking Right"

                        elif gaze.is_left():
                            self.person[i][1]+=1
                            self.text = f"Person #{i} Looking Left"
                            
                        elif gaze.is_center():
                            self.person[i][0]+=1
                            self.text = f"Person #{i} Looking Center"

                        attention = "Attention Score: " + str(round((self.person[i][0] / len(detected_frames)) * 100 ,2)) + " %"

                        try:
                            cv2.putText(frame, self.text, (x1-40, y-10),cv2.LINE_AA, 1, (0, 255, 0), 3)
                            cv2.putText(frame, attention, (x1-40, y-75),cv2.LINE_AA, 1, (0, 255, 0), 3)
                        except:
                            continue


                FlippedImage = frame
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1200, 900, Qt.KeepAspectRatio)

                self.ImageUpdate.emit(Pic)

            print(self.person)
            with open('results.pkl', 'wb') as f:
                pickle.dump(self.person, f)

        

            
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())