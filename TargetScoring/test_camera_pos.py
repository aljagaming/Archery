import cv2

class TestCameraPos:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        while True:
            ret, frame = self.video.read()

            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = TestCameraPos()