
'''
author : "Amit Yadav"
email : "amitech90@gmail.com"
github : "[warriorwizard](https://github.com/warriorwizard)"

'''

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('wrist_estimator_10_epochs.h5')

cap = cv2.VideoCapture(0)

frame_counter = 0  

while cap.isOpened():
    _ , frame = cap.read()
    
    if frame_counter == 0:  # Process this frame
        # rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame, (250,250))
        
        coords = model.predict(np.expand_dims(resized/255,0))[0]

        for i in range(0, len(coords), 2):
            x = coords[i]
            y = coords[i+1]
            cv2.circle(resized, ((int(x), int(y))), 3, (255, 255, 0), -1)

        cv2.imshow('frame', resized)

    # Increment frame_counter or reset to 0 once it reaches 5
    frame_counter = (frame_counter + 1) % 2
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
