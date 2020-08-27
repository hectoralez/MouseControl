# MouseControl
Run multiple pretrained models in the OpenVINO toolkit to control your computer pointer using your eye gaze.

This project allows for control over the computer pointer by running inference on a video or through a web cam. Four pre-trained models are used together to give the correct feedback to move the computer pointer.

## Project Set Up and Installation
Download:
-OpenVino 2020
-python3 requirements
```
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```
-models
```
sudo ./downloader.py --name face-detection-adas-binary-0001 -o /desktop/starter
sudo ./downloader.py --name head-pose-estimation-adas-0001 --precision=FP16,FP32 -o /desktop/starter
sudo ./downloader.py --name landmarks-regression-retail-0009 --precision=FP16,FP32 -o /desktop/starter
sudo ./downloader.py --name gaze-estimation-adas-0002 --precision=FP16,FP32 -o /desktop/starter
```

It is recommended to put models close by to shorten model paths.

--input bin/demo.mp4 or cam
--device CPU or [CPU, GPU, MYRIAD, FPGA]
--face_model intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001
--pose_model intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
--landmarks_model intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009
--gaze_model intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
--precision FP16 or FP32
--display True or False
--output output

## Demo
```
#This will display a description of the input arguments.
python3 src/main.py --help

#run with webcam and FP32. display on
python3 src/main.py --input cam \
--device CPU \
--face_model intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 \
--pose_model intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
--landmarks_model intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
--gaze_model intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 \
--precision FP32 \
--display \
--output output

#run with cam and FP16
python3 src/main.py --input cam \
--device CPU \
--face_model intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 \
--pose_model intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 \
--landmarks_model intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 \
--gaze_model intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 \
--precision FP16 \
--display \
--output output

#run with demo video and FP32. Display included
python3 src/main.py --input bin/demo.mp4 \
--device CPU \
--face_model intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 \
--pose_model intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
--landmarks_model intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
--gaze_model intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 \
--precision FP32 \
--display \
--output output

#run with demo video and FP16. Display not included
python3 src/main.py --input bin/demo.mp4 \
--device CPU \
--face_model intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 \
--pose_model intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 \
--landmarks_model intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 \
--gaze_model intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 \
--precision FP16 \
--output output
```

## Documentation
There are 6 files that are used by main.py to make the app work.
1. input_feeder.py handles the video input and creates a stream of frames
2. mouse_controller.py waits for x y coordinates to move the computer pointer on its own.
3. face_detection.py loads the model face-detection-adas-binary-0001. Inputs an image. Outputs cropped faces detected.
4. head_pose_estimation.py loads the model head-pose-estimation-adas-0001. Inputs a cropped face. Outputs Eulers angles (yaw, pitch, roll)
5. facial_landmarks_detection.py loads the model landmarks-regression-retail-0009. Inputs a cropped face. Outputs 10 coordinates for 5 facial landmarks.
6. gaze_estimation.py loads the model gaze-estimation-adas-0002. Inputs a cropped face, facial landmarks and pose angles. Outputs a vector in cartesian coordinates (x,y,z). X and Y can be used in mouse_controller.py to move the computer pointers position.

## Benchmarks
My app was run with FP32 and FP16 model. I have kept track of model load time, input preprocessing time, inference time, and output preprocessing time.

model | device | precision |  load  |  input   | inference | output
------|--------|-----------|--------|----------|-----------|---------
Face  |  cpu   | FP32/INT1 | 0.3431 | 0.000685 | 0.010709  | 0.000508
Pose  |  cpu   | FP32      | 0.0840 | 5.88e-05 | 0.001609  | 1.44e-05
Land  |  cpu   | FP32      | 0.0575 | 4.70e-05 | 0.000513  | 8.50e-05
Gaze  |  cpu   | FP32      | 0.1101 | 3.16e-05 | 0.001796  | 4.37e-06
Pose  |  cpu   | FP16      | 0.0893 | 5.93e-05 | 0.001614  | 1.59e-05
Land  |  cpu   | FP16      | 0.0517 | 4.88e-05 | 0.000512  | 8.16e-05
Gaze  |  cpu   | FP16      | 0.1090 | 3.23e-05 | 0.001803  | 3.99e-06


## Results
I noticed that FP32 precision takes a bit longer but nothing major. I also wanted to create a job submission script to work with other hardware but I ran out of time.

I also had a better result and control over the mouse by changing the speed to a decimal. A second or greater is too slow for feedback.

I played around with multiplying negative one to the mouse controler x and y coordinates. This will invert the controls over the mouse.

## Stand Out Suggestions
1. Change in precision to improve performance.
2. Benchmark running times but I do not have a toggle option.
3. Edge cases like multiple people and skipping over bad inference frames.
4. Video and webcam capability.
5. Display flag to allow for outputs of intermediate models


### Edge Cases
1. If no face is detected I skipped the gaze estimation so that the mouse class would not break to bad x,y coordinates.
2. For multiple faces I focused on the largest face since they would be closest to the camera. So only one person should control the mouse.
3. Allow for user to use a webcam or video file. Allow for precisions of FP16 and FP32.
4. Bad lighting such as covering the camera with your finger causes the mouse to be centered so that control can go back to normal when inference is detected.
