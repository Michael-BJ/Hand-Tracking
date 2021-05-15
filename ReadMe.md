# How it's works
1. First import the library that we need
````
import cv2
import numpy as np
import mediapipe as mp
````
2. Make the program to connect to the webcam
```
import cv2
import numpy as numpy

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
`````
3. Load the module of hands and drawing_utils
````
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
````
4. Determine the minimum percentage
````
with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
````
5. Chanfe BGR to RGB
````
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
````
6. To optimize the program change writeable to False
````
image.flags.writeable = False
````
7. processing
````
results = hands.process(image)
````
8. Change RGB to BGR 
````
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
````
9. Make a loop and draw the landmark 
````
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            	image, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(0, 0, 255 )),
                mp_drawing.DrawingSpec(color=(255, 255, 255 )))
````                                        