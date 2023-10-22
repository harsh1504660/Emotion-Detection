import onnxruntime as rt
import cv2
import numpy as np
import time
import service.main as s

def emotions_detector(img_array):
    time_init = time.time()

    test_image = cv2.resize(img_array, (256,256))

    im = np.float32(test_image)
    img_arr = np.expand_dims(im,axis=0)

    onnx_pred = s.m_q.run(['dense_2'],{'input':img_arr})
    time_elapsed = time.time() - time_init
    pred = np.argmax(onnx_pred[0][0])

    emotion=''
    if pred ==0:
        emotion='angry'
    elif pred==1:
        emotion='happy'
    else:
        emotion='sad'

    return {
        "emotion":emotion,
        "time_elapsed":str(time_elapsed)
        }