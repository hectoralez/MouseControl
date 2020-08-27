'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import numpy as np
import time
from model import Model

class FacialLandmarksDetectionModel(Model):
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, output, device='CPU'):
        self.tmlt_landmarks = super().__init__(model_name, device)

    def predict(self, image):
        height, width, channels = image.shape
        start_input_time=time.time()
        processed_img = self.preprocess_input(image)
        tinpt_landmarks = time.time() - start_input_time
        input_dict={self.input_name:[processed_img]}
        start_inference_time=time.time()
        result_infer = self.net.infer(input_dict)
        tint_landmarks = time.time() - start_inference_time
        start_output_time=time.time()
        landmarks = self.preprocess_output(result_infer, height, width)
        toutt_landmarks = time.time() - start_output_time
        return landmarks, self.tmlt_landmarks, tinpt_landmarks, tint_landmarks, toutt_landmarks

    def preprocess_output(self, outputs, height, width):
        landmarks = []
        out = outputs[self.output_name]
        detection = np.squeeze(out[0])
        xleft_eye = int(detection[0] * width)
        yleft_eye = int(detection[1] * height)
        xright_eye = int(detection[2] * width)
        yright_eye = int(detection[3] * height)
        xnose = int(detection[4] * width)
        ynose = int(detection[5] * height)
        xleft_mouth = int(detection[6] * width)
        yleft_mouth = int(detection[7] * height)
        xright_mouth = int(detection[8] * width)
        yright_mouth = int(detection[9] * height)
        landmark = [xleft_eye, yleft_eye, xright_eye, yright_eye, xnose, ynose, xleft_mouth, yleft_mouth, xright_mouth, yright_mouth]
        return landmark
