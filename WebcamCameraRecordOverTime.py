import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)
cap.set(cv2.CAP_PROP_AUTO_WB,0)
cap.set(cv2.CAP_PROP_GAMMA,1)
size_x = 1280
size_y = 640
center = (int(np.round(size_x / 2)), int(np.round(size_y / 2)))
print(center[0], center[1])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_x)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_y)
array = []
recording = False

while True:
    ret, frame = cap.read()
    pixel = frame[center[1],center[0]]
    b,g,r=map(int,pixel)
    cv2.rectangle(frame, (center[0]-100, center[1]-100), (center[0]+100, center[1]+100), (b,g,r), -1)
    cv2.imshow('frame',frame)
    if recording:
        array.append((b,g,r))
        cv2.putText(frame,"Rec",(20,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        recording = not recording
        print(recording)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(type)
intensity = []
index = []
for i in range(len(array)):
    intensity.append(array[i][0]+array[i][1]+array[i][2])
    index.append(i)

plt.plot(index,intensity,color='blue',marker='o')
plt.show()
