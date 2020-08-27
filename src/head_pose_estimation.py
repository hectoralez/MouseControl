'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import numpy as np
import time
from model import Model


class HeadPoseEstimationModel(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, output, device='CPU'):
        self.tmlt_pose = super().__init__(model_name, device)

    def predict(self, image):
        start_input_time=time.time()
        processed_img = self.preprocess_input(image)
        tinpt_pose = time.time() - start_input_time
        input_dict={self.input_name:[processed_img]}
        start_inference_time=time.time()
        result_infer = self.net.infer(input_dict)
        tint_pose = time.time() - start_inference_time
        start_output_time=time.time()
        pose = self.preprocess_output(result_infer)
        toutt_pose = time.time() - start_output_time
        return pose, self.tmlt_pose, tinpt_pose, tint_pose, toutt_pose

    def preprocess_output(self, outputs):
        yaw = np.squeeze(outputs["angle_y_fc"])
        pitch = np.squeeze(outputs["angle_p_fc"])
        roll = np.squeeze(outputs["angle_r_fc"])
        processed_output = [yaw, pitch, roll]
        return processed_output
