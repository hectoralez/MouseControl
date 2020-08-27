import argparse
import cv2
import os
import numpy as np
import logging

from face_detection import FaceDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bin/demo.mp4", type=str,
                        help="Path to image or video file")
    parser.add_argument("--device", default="CPU", type=str,
                        help="Device to run inference (CPU, GPU, FPGA, MYRIAD)")
    parser.add_argument("--face_model", default="intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001", type=str,
                        help="Path to face detection model")
    parser.add_argument("--pose_model", default="intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001", type=str,
                        help="Path to pose estimation model ")
    parser.add_argument("--landmarks_model", default="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009", type=str,
                        help="Path to landmarks detection model")
    parser.add_argument("--gaze_model", default="intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002", type=str,
                        help="Path to gaze estimation model")
    parser.add_argument("--precision", default="FP32", type=str,
                        help="Precision being used (FP32, FP16)")
    parser.add_argument("--display", default=False, action="store_true",
                        help="Flag that can display the outputs of intermediate models")
    parser.add_argument("--output", default="/output", type=str,
                        help="Path to output")
    return parser

def main():
    args = build_argparser().parse_args()
    logging.basicConfig(filename=args.output+'/app.log', filemode='w')

    print("Begin: Try not to move mouse with your hands")
    mc = MouseController("low", "fast")
    if args.input == "cam":
        frames = InputFeeder("cam")
    else:
        frames = InputFeeder("video", args.input)
    cap = frames.load_data()

    if args.display:
        initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_video = cv2.VideoWriter(os.path.join(args.output, 'output_video.mp4'), cv2.VideoWriter_fourcc('m','p','4','v'), fps, (initial_w, initial_h))


    face_model = FaceDetectionModel(args.face_model, args.output, args.device)
    pose_model = HeadPoseEstimationModel(args.pose_model, args.output, args.device)
    landmarks_model = FacialLandmarksDetectionModel(args.landmarks_model, args.output, args.device)
    gaze_model = GazeEstimationModel(args.gaze_model, args.output, args.device)
    avg_out = 0
    avg = 0
    tmlt_face_avg = 0
    tinpt_face_avg = 0
    tint_face_avg = 0
    toutt_face_avg = 0

    tmlt_pose_avg = 0
    tinpt_pose_avg = 0
    tint_pose_avg = 0
    toutt_pose_avg = 0

    tmlt_landmarks_avg = 0
    tinpt_landmarks_avg = 0
    tint_landmarks_avg = 0
    toutt_landmarks_avg = 0

    tmlt_gaze_avg = 0
    tinpt_gaze_avg = 0
    tint_gaze_avg = 0
    toutt_gaze_avg = 0
    logging.info("Frames starting")
    for frame in frames.next_batch():
        if frame is None:
            logging.error("Frame: " + frame + "failed")
            continue
        output_image = frame.copy()
        cropped_faces, tmlt_face, tinpt_face, tint_face, toutt_face = face_model.predict(frame)
        try:
            largest_face = cropped_faces[0]
            for face in cropped_faces:
                if largest_face.size < face.size:
                    largest_face = face
            pose, tmlt_pose, tinpt_pose, tint_pose, toutt_pose = pose_model.predict(largest_face)
            landmarks, tmlt_landmarks, tinpt_landmarks, tint_landmarks, toutt_landmarks = landmarks_model.predict(largest_face)
            gaze_vector, tmlt_gaze, tinpt_gaze, tint_gaze, toutt_gaze = gaze_model.predict(largest_face, landmarks, pose)
        except Exception as e:
            logging.error("Model inference failed: " + str(e))
            # print(e)
            continue
        if args.display:
            output_image, xmin, ymin = face_model.draw_crop_outputs(output_image, args.display)
            output_image = gaze_model.display_eye_boxes(output_image, landmarks, xmin, ymin, args.display)
            out_video.write(output_image)
        cv2.imshow("output_image", output_image)
        cv2.waitKey(15)
        face_model.coords = []
        tmlt_face_avg += tmlt_face
        tinpt_face_avg += tinpt_face
        tint_face_avg += tint_face
        toutt_face_avg += toutt_face

        tmlt_pose_avg += tmlt_pose
        tinpt_pose_avg += tinpt_pose
        tint_pose_avg += tint_pose
        toutt_pose_avg += toutt_pose

        tmlt_landmarks_avg += tmlt_landmarks
        tinpt_landmarks_avg+= tinpt_landmarks
        tint_landmarks_avg += tint_landmarks
        toutt_landmarks_avg += toutt_landmarks

        if gaze_vector is None:
            avg_out += 1
            continue
        tmlt_gaze_avg += tmlt_gaze
        tinpt_gaze_avg += tinpt_gaze
        tint_gaze_avg += tint_gaze
        toutt_gaze_avg += toutt_gaze
        avg += 1
        gaze_vector_norm = gaze_vector / np.linalg.norm(gaze_vector)
        try:
            mc.move(gaze_vector_norm[0], gaze_vector_norm[1])
        except Exception as e:
            logging.error("Gaze failed: " + str(e))
            # print(e)
            continue

    file_name = "stats_"+args.precision+".txt"
    save_path = os.path.join(os.getcwd(), args.output)
    f = open(os.path.join(save_path, file_name), "w")
    f.write("Benchmark Start:"+"\n\n")
    f.write("Face Detection Model stats"+"\n")
    f.write("Total model Load Time:"+str(tmlt_face_avg/avg)+"\n")
    f.write("Total Input Time:"+str(tinpt_face_avg/avg)+"\n")
    f.write("Total Inference Time:"+str(tint_face_avg/avg)+"\n")
    f.write("Total Output Time:"+str(toutt_face_avg/avg)+"\n\n")

    f.write("Head Pose Estimation Model stats"+"\n")
    f.write("Total model Load Time:"+str(tmlt_pose_avg/avg)+"\n")
    f.write("Total Input Time:"+str(tinpt_pose_avg/avg)+"\n")
    f.write("Total Inference Time:"+str(tint_pose_avg/avg)+"\n")
    f.write("Total Output Time:"+str(toutt_pose_avg/avg)+"\n\n")

    f.write("Facial Landmarks Detection Model stats"+"\n")
    f.write("Total model Load Time:"+str(tmlt_landmarks_avg/avg)+"\n")
    f.write("Total Input Time:"+str(tinpt_landmarks_avg/avg)+"\n")
    f.write("Total Inference Time:"+str(tint_landmarks_avg/avg)+"\n")
    f.write("Total Output Time:"+str(toutt_landmarks_avg/avg)+"\n\n")

    f.write("Gaze Estimation Model stats"+"\n")
    f.write("Total model Load Time:"+str(tmlt_gaze_avg/(avg-avg_out))+"\n")
    f.write("Total Input Time:"+str(tinpt_gaze_avg/(avg-avg_out))+"\n")
    f.write("Total Inference Time:"+str(tint_gaze_avg/(avg-avg_out))+"\n")
    f.write("Total Output Time:"+str(toutt_gaze_avg/(avg-avg_out))+"\n\n")
    f.write("Benchmark end"+"\n")
    f.close()

    print("Thank you, Goodbye")
    frames.close()

if __name__ == '__main__':
    main()
