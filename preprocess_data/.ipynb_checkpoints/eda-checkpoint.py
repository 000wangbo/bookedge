import cv2
import os 
import numpy as np
for i in os.listdir('/data/jupyter/bookedge/train_datasets_document_detection_0411/segments/'):
    img = cv2.imread('/data/jupyter/bookedge/train_datasets_document_detection_0411/segments/' + i)
    if len(np.unique(img)) > 2:
        print('asdasdads')
