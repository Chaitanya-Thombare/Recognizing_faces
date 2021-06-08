#Importing Libraries
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

model = keras.models.load_model(r'C:\Users\indct\Desktop\Workspace\Projects\OpenCV\models\trained.h5')
vid = cv.VideoCapture(0)

ret = True
while ret:
    ret, frame = vid.read()
    
    temp_frame = cv.resize(frame, (180, 180))
    temp_frame = cv.cvtColor(temp_frame, cv.COLOR_BGR2GRAY)

    img_pred_nparr = image.img_to_array(temp_frame)
    img_pred_nparr = np.expand_dims(img_pred_nparr,axis = 0)

    result = model.predict_classes(img_pred_nparr)
    
    if result == 0:
        pred_class = "Chaitu"
    else:
        pred_class = "Dada"
    
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (100, 50)
    fontScale = 3
    color = (255, 0, 0)
    thickness = 2
    frame = cv.putText(frame, pred_class, org, font, 
                    fontScale, color, thickness, cv.LINE_AA)

    cv.imshow("Live Video", frame)

    if cv.waitKey(10) & 0xff == ord('q'):
        break

vid.release()
cv.destroyAllWindows()