import time

import cv2
import subprocess
import threading

from evaluator import Evaluator


class Detector:
    def __init__(self, camera_number):
        self.camera_number = camera_number
        self.time_between_checks_ms = None
        self.evaluator = None
        self.currently_detecting = True;
        self.pixel_threshold = 50



    def detect(self):


        video = cv2.VideoCapture(self.camera_number)
        video.set(cv2.CAP_PROP_EXPOSURE, 2)

        print(video.get(cv2.CAP_PROP_EXPOSURE))

        THRESHOLD = 0.1

        ret, frame = video.read()

        if not ret:
            print("FIRST camera read failed")
            return
        else:
            self.evaluator = Evaluator(frame)
            print("First camera read sucesfully evaluator instantiated ")

        print("Warming up camera...")
        for _ in range(30):  # skip 30 frames (~1 second)
            video.read()
        for _ in range(30):  # skip 30 frames (~1 second)
            video.read()

        ret, frame_old = video.read()
        ret, frame_new = video.read()


        while self.currently_detecting:

            frame_old = frame_new
            ret, frame_new = video.read()

            if not ret:
                print("ERROR: Camera read failed")
                continue


            #LOTS OF FLICKERing HERE
            frame_old_blur = cv2.GaussianBlur(frame_old, (21, 21), 0)
            frame_new_blur = cv2.GaussianBlur(frame_new, (21, 21), 0)

            diff = cv2.absdiff(frame_old_blur, frame_new_blur)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)        #if pixel changes value more then 30
                                                                                             # returns the treshold and the thresh filtered output
            diff_pixels_num = cv2.countNonZero(thresh)

            print(f"Number of different pixels: {diff_pixels_num}")



            if diff_pixels_num > self.pixel_threshold:
                print("CHANGEEDDDDD!!!!!")
                threading.Thread(target=play_sound).start()
                cv2.imshow("The picture", frame_new)
                cv2.waitKey(0)
                score = self.evaluator.evaluate(frame_new)
                print(f"Changed detected; score is: {score} at time: {time.time()}")
                self.evaluator.debug();
                break
            else:
                print("nothing changed!!!!!")




def play_sound():
    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])