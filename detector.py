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

        self.video = cv2.VideoCapture(camera_number)
        self.video.set(cv2.CAP_PROP_EXPOSURE, 2)


    def detect(self):

        ret, frame = self.video.read()
        if not ret:
            print("FIRST camera read failed")
            return

        print("Warming up camera...")
        for _ in range(30):  # skip 30 frames (~1 second)
            self.video.read()
        for _ in range(30):  # skip 30 frames (~1 second)
            self.video.read()

        print("Initialising evaluator...")
        n_frames = 200;
        frame_accumulation = self.frame_accumulator(n_frames)
        self.evaluator = Evaluator(frame_accumulation)
        print("Evaluator initialized successfully!")

        # Change detection ---------------------------------------------------
        ret, frame_new = self.video.read()
        while self.currently_detecting:

            frame_old = frame_new
            ret, frame_new = self.video.read()

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


    def frame_accumulator (self, number_of_frames):
        frame_acumulation = [];

        for i in range(number_of_frames):
            ret, frame = self.video.read()
            if ret:
                frame_acumulation.append(frame)
        return frame_acumulation


def play_sound():
    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])