import bisect
import cv2
import json
from pathlib import Path

class Processor:
    def __init__(self, input_folder, arrow_score_timestamp):

        self.video_length_sec = 5
        self.FPS = 24.0

        self.root_folder = Path(input_folder)
        self.arrow_score_timestamp = arrow_score_timestamp      #Contains [(timestamp_of_arrow_hit, score)]

        with open(self.root_folder / "front_timestamps.json") as f:self.front_timestamps = json.load(f)
        with open(self.root_folder / "back_timestamps.json") as f:self.back_timestamps = json.load(f)

        self.front_video_path = str(self.root_folder / "front.mp4")
        self.back_video_path = str(self.root_folder / "back.mp4")

        cap = cv2.VideoCapture(self.front_video_path)
        self.FPS = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def cutVideos(self):
        output_dir = self.root_folder / "CutVideos"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (timestamp, score) in enumerate(self.arrow_score_timestamp):
            for video_path, timestamps, side in [
                (self.front_video_path, self.front_timestamps, "front"),
                (self.back_video_path, self.back_timestamps, "back"),
            ]:
                frames = self.extractFrames(video_path, timestamps, timestamp)
                out_path = output_dir / f"ARROW{i + 1}_{score}_{side}.mp4"
                self.writeClip(frames, str(out_path))
                print(f"Saved {out_path}")

        print("Videos cut")

    def extractFrames(self, video_path, timestamps, trigger_timestamp):
        start_index = self.binarySearch(timestamps, trigger_timestamp)
        end_time = trigger_timestamp + self.video_length_sec
        end_index = self.binarySearch(timestamps, end_time)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        #cap.set(self.FPS, start_index)
        print(cv2.CAP_PROP_POS_FRAMES)

        frames = []
        for _ in range(end_index - start_index):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def writeClip(self, frames, out_path):
        if not frames:
            print(f"No frames for {out_path}")
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, writer, self.FPS, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()

    def binarySearch(self, timestamp_array, target):
        i = bisect.bisect_left(timestamp_array, target)
        if i == 0:
            return 0
        if i == len(timestamp_array):
            return len(timestamp_array) - 1
        if timestamp_array[i] - target < target - timestamp_array[i - 1]:
            return i
        return i - 1



if __name__ == '__main__':
    print("Hello")
    output_folder = "/Users/al/Desktop/Recordings/Raw/17-06-2026/17-40-36/"
    arrow_score_timestamp_test = [
        (1781710843.0, 9),
        (1781710849.0, 9),
        (1781710854.0, 8),
    ]
    processor = Processor(output_folder,arrow_score_timestamp_test)
    processor.cutVideos()

