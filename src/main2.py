import cv2

vs = cv2.VideoCapture(0)

while True:
    ret,frame = vs.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
