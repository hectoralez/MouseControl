'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import time
import cv2
from model import Model

class GazeEstimationModel(Model):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, output, device='CPU'):
        self.tmlt_gaze = super().__init__(model_name, device)

    def predict(self, cropped_face, landmarks, pose):
        start_input_time=time.time()
        processed_input = self.preprocess_input(cropped_face, landmarks, pose)
        tinpt_gaze = time.time() - start_input_time
        if processed_input is None:
            gaze_vector = None
            tint_gaze = 0
            toutt_gaze = 0
            return gaze_vector, self.tmlt_gaze, tinpt_gaze, tint_gaze, toutt_gaze
        start_inference_time=time.time()
        result_infer = self.net.infer(processed_input)
        tint_gaze = time.time() - start_inference_time
        start_output_time=time.time()
        gaze_vector = self.preprocess_output(result_infer)
        toutt_gaze = time.time() - start_output_time
        return gaze_vector, self.tmlt_gaze, tinpt_gaze, tint_gaze, toutt_gaze

    def preprocess_input(self, cropped_face, landmarks, pose):
        xleft_eye = landmarks[0]
        yleft_eye = landmarks[1]
        xright_eye = landmarks[2]
        yright_eye = landmarks[3]
        left_eye_image = self.crop_eyes(cropped_face, xleft_eye, yleft_eye)
        right_eye_image = self.crop_eyes(cropped_face, xright_eye, yright_eye)
        if left_eye_image is None or right_eye_image is None:
            return None
        head_pose_angles = pose
        processed_input = {"left_eye_image": left_eye_image,
                            "right_eye_image": right_eye_image,
                            "head_pose_angles": head_pose_angles}
        return processed_input

    def display_eye_boxes(self, image, landmarks, xmin, ymin, display=False):
        xleft_eye = landmarks[0] + xmin
        yleft_eye = landmarks[1] + ymin
        xright_eye = landmarks[2] + xmin
        yright_eye = landmarks[3] + ymin
        cv2.rectangle(image, (xleft_eye-30, yleft_eye-30), (xleft_eye+30, yleft_eye+30), (0, 55, 255), 2)
        cv2.rectangle(image, (xright_eye-30, yright_eye-30), (xright_eye+30, yright_eye+30), (0, 55, 255), 2)
        return image

    def crop_eyes(self, cropped_face, x, y, width=60, height=60):
        ymin = int(y - height/2)
        ymax = int(y + height/2)
        xmin = int(x - width/2)
        xmax = int(x + width/2)
        cropped_eye = cropped_face[ymin:ymax, xmin:xmax, :].copy()
        if cropped_eye.shape[0] != height or cropped_eye.shape[1] !=width:
            return None
        processed_img = cropped_eye.transpose((2, 0, 1))
        processed_img = processed_img.reshape(1, 3, 60, 60)
        return processed_img

    def preprocess_output(self, outputs):
        gaze_vector = outputs["gaze_vector"][0]
        return gaze_vector
