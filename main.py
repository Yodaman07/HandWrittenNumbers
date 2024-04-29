import cv2 as cv
import os
import cv2.typing


def getLatestSave():  # Finds the last saved image for assigning IMG_INDEX
    return len(os.listdir("saves"))


class IMG:
    def __init__(self):
        self.IMG_INDEX = getLatestSave()
        if not os.path.exists("saves"):
            os.mkdir("saves")

    def save(self, img: cv2.typing.MatLike):  # Saves an image
        cv.imwrite(f"saves/SAVE_{str(self.IMG_INDEX)}.png", img)
        self.IMG_INDEX += 1


cam = cv.VideoCapture(0)  # makes videocapture object --> have to release it afterward
i = IMG()
if not cam.isOpened():
    print("Unable to access camera")  # kill the program if the camera is not accessed
    cam.release()
    exit()

while True:
    retrieved, frame = cam.read()
    if not retrieved:
        print("Stream has likely ended")
        break

    cv.imshow("stream", frame)
    # https://stackoverflow.com/questions/5217519/what-does-opencvs-cvwaitkey-function-do <-- how waitKey works
    key = cv.waitKey(1)
    if key == ord("q"):  # gets the unicode value for q
        break
    elif key == ord("s"):
        i.save(frame)

cam.release()
cv.destroyAllWindows()
