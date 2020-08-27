'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
from model import Model


class FaceDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, output, device='CPU', threshold=0.5):
        self.tmlt_face = super().__init__(model_name, device)
        self.coords = []
        self.threshold = threshold

    def predict(self, image):
        height, width, channels = image.shape
        start_input_time=time.time()
        processed_img = self.preprocess_input(image)
        tinpt_face = time.time() - start_input_time
        input_dict={self.input_name:[processed_img]}
        start_inference_time=time.time()
        result_infer = self.net.infer(input_dict)
        tint_face = time.time() - start_inference_time
        start_output_time=time.time()
        coords = self.preprocess_output(result_infer, height, width)
        toutt_face = time.time() - start_output_time
        cropped_faces = self.draw_crop_outputs(image)
        return cropped_faces, self.tmlt_face, tinpt_face, tint_face, toutt_face

    def draw_crop_outputs(self, image, display=False):
        boxed_image = image.copy()
        cropped_faces = []
        for coord in self.coords:
            xmin = coord[0]
            ymin = coord[1]
            xmax = coord[2]
            ymax = coord[3]
            if display:
                cv2.rectangle(boxed_image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 2)
                return boxed_image, xmin, ymin
            cropped_face = image[ymin:ymax, xmin:xmax, :].copy()
            cropped_faces.append(cropped_face)
        return cropped_faces

    def preprocess_output(self, outputs, height, width):
        out = outputs[self.output_name]
        for detection in out[0][0]:
            if detection[2] > self.threshold:
                xmin = int(detection[3] * width)
                ymin = int(detection[4] * height)
                xmax = int(detection[5] * width)
                ymax = int(detection[6] * height)

                coord = [xmin, ymin, xmax, ymax]
                self.coords.append(coord)
        return self.coords
