import time
from time import sleep

import cv2
import subprocess
import threading
from .evaluator import Evaluator


class Detector:
    def __init__(self, camera_number):
        self.camera_number = camera_number

        self.currently_detecting = True;
        self.pixel_threshold = 50
        self.fps = 23                   #tested
        self.detection_rate_delay = 0
        self.arrow_number = 1

        self.video = cv2.VideoCapture(camera_number)


    def detect(self):

        ret, frame = self.video.read()
        if not ret:
            print("FIRST camera read failed")
            return

        print("Warming up camera...")
        for _ in range(30):  # skip 30 frames (~1 second)
            self.video.read()

        print("Initialising evaluator...")
        n_frames = 20;
        frame_accumulation = self.frame_accumulator(n_frames)
        self.evaluator = Evaluator(frame_accumulation)
        print("Evaluator initialized successfully!")

        # Change detection ---------------------------------------------------
        print("Detecting...")
        ret, frame_new = self.video.read()

        test_counter=0;

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

            #print(f"Number of different pixels: {diff_pixels_num}")



            if diff_pixels_num > self.pixel_threshold:
                time_of_shot = time.time()
                print("Change detected!")
                threading.Thread(target=play_sound).start()
                self.evaluator.add_stage("Detector: Detected change image:",frame_new)
                score = self.evaluator.evaluate(frame_new)
                print(f"Changed detected; score is: {score} at time: {time_of_shot}")
                test_counter += 1

                #self.evaluator.debug();
            if (test_counter >= self.arrow_number):
                self.currently_detecting = False


            sleep(self.detection_rate_delay)
        self.evaluator.debug();



    def frame_accumulator (self, number_of_frames):
        buffer = [];

        for i in range(number_of_frames):
            ret, frame = self.video.read()
            if ret:
                buffer.append(frame)
        return buffer


def play_sound():
    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])