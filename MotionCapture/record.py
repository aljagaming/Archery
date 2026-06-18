import threading
import time
from time import sleep
import cv2
import json
from datetime import datetime
from pathlib import Path

class Recorder:
    def __init__(self):
        self.root_folder = "/Users/al/Desktop/Recordings/Raw/"

        self.front_facing_camera = 1
        self.back_facing_camera = 2

        self.FPS = 24


        #self.processor = Processor(f"{self.output_folder}/front.mp4",
        #                           f"{self.output_folder}/back.mp4",
        #                           self.front_timestamps,
        #                           self.back_timestamps,
        #                           self.FPS)



    def startRecording(self):
        self.recording = True

        self.video_front = cv2.VideoCapture(self.front_facing_camera)
        self.video_back = cv2.VideoCapture(self.back_facing_camera)

        self.threads = []
        self.front_timestamps = []
        self.back_timestamps = []
        self.output_folder = self.dateDirectory()  # fresh folder per session

        width_front = int(self.video_front.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_front = int(self.video_front.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_back = int(self.video_back.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_back = int(self.video_back.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.write_lock = threading.Lock()

        writer = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_front = cv2.VideoWriter(f"{self.output_folder}/front.mp4", writer, 24, (width_front, height_front))
        self.out_back = cv2.VideoWriter(f"{self.output_folder}/back.mp4", writer, 24, (width_back, height_back))

        self.front_thread = threading.Thread(
            target=self.capture,
            args=(self.video_front, self.front_timestamps, self.out_front, self.FPS),
            name="capture_front",
            daemon=True
        )

        self.back_thread = threading.Thread(
            target=self.capture,
            args=(self.video_back, self.back_timestamps, self.out_back, self.FPS),
            name="capture_back",
            daemon=True
        )

        self.threads.append(self.front_thread)
        self.threads.append(self.back_thread)

        self.front_thread.start()
        self.back_thread.start()

    def stopRecording(self):
        self.recording = False
        for t in self.threads:
            t.join()
        self.out_front.release()
        self.out_back.release()
        #writes to jason but doesnt really have to as it is in the array already so just pass it to the thing
        with open(f"{self.output_folder}/front_timestamps.json", "w") as f:json.dump(self.front_timestamps, f)
        with open(f"{self.output_folder}/back_timestamps.json", "w") as f:json.dump(self.back_timestamps, f)

        print("Recording finished! ")


        #random_timestamp = self.front_timestamps[len(self.front_timestamps) // 3]
        #self.processor.trigger_timestamps = [random_timestamp]
        #self.processor.process()


    def capture(self, video, timestamp_array, output, FPS):
        # very complicated thing
        # parallely start all the cameras so the least delay possible
        # but then writer needs to be thread safe so use self.write_lock
        # also check that camera is not recording faster then the FPS

        while self.recording:
            ret, frame = video.read()
            if ret:
                timestamp_array.append(time.time())
                with self.write_lock:
                    output.write(frame)


    def dateDirectory(self):
        session_time = datetime.now().strftime("%d-%m-%Y/%H-%M-%S")
        directory = Path(self.root_folder) / session_time
        directory.mkdir(parents=True, exist_ok=True)
        return directory



if __name__ == '__main__':
    record = Recorder()


    for i in range(2):
        print("This is loop iteration:", i)
        record.startRecording()
        sleep(20)
        record.stopRecording()
