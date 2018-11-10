from styx_msgs.msg import TrafficLight
import cv2
import csv

class ImageSaver(object):
    def __init__(self):
        self.counter = 0

    def save_image(self, image, color):
        #image file save
        image = image[0:400,250:550]

        cv2.imwrite('./light_classification/img/'+ str(self.counter) +'.jpg', image)
        self.counter += 1

        #result file save
        data = [self.counter, color]
        with open('./light_classification/result.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(data)
