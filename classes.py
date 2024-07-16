import cv2
import numpy as np
# from tensorflow.keras.models import load_model
# from keras.src.legacy.saving import legacy_h5_format
from ultralytics import YOLO
import copy

class PlayersDetector:
    def __init__(self,input_video_path):
        self.input_video_path=input_video_path
    def players_detection(self,model_path):
        model=YOLO(model_path)
        cap=cv2.VideoCapture(self.input_video_path)
        output_video_frames=[]
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            prediction=model.predict(frame)
            # print('box:',prediction[0].boxes)
            # print('xywh:',prediction[0].boxes.xywh)
            data=prediction[0].boxes.data
            for i in range(len(data)):
                x_min,y_min,x_max,y_max=data[i][:4]
                conf=data[i][4]
                cls=data[i][5]
                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),2)
                output_video_frames.append(frame)
detect=PlayersDetector( r"D:\tennis_yolo_video\input_video.mp4")
detect.players_detection('D:\Tennis_yolo\yolov8x.pt')














#
#
#
# model = legacy_h5_format.load_model_from_hdf5(r'D:\Tennis_yolo\tennis_keypoints_detector\new_model.h5', custom_objects={'mse': 'mse'})
#
# # Load the pre-trained model
# #model = load_model(r'D:\Tennis_yolo\tennis_keypoints_detector\better_tennis_checkpoints_detector.h5')
#
# # Video input path
# input_video_path = r"D:\tennis_yolo_video\input_video.mp4"
#
# # Video output path
# output_video_path = r'"D:\video_folder"\newest_video3.avi'
#
# # Open the input video
# cap = cv2.VideoCapture(input_video_path)
#
# # Get video properties
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print('basic:',frame_width,frame_height,fps,total_frames)
#
# # Initialize video writer
# out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
#
# # Process each frame
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Perform any preprocessing or model-specific processing here
#     # Example: Normalize frame to [0, 1] range and convert to float32
#     frame = frame.astype(np.float32) / 255.0
#
#     # Example: Resize frame to match model input size (224x224), if needed
#     frame_resized = cv2.resize(frame, (224, 224))
#
#     # Example: Predict using the model
#     # Note: Adjust this part according to your model's input and output requirements
#     prediction = model.predict(np.expand_dims(frame_resized, axis=0))
#
#     # Example: Post-processing the prediction to draw keypoints on the frame
#     # Adjust this according to your model's output format
#     keypoints = prediction.reshape(-1, 2)  # Assuming prediction is a flattened array of (x, y) coordinates
#
#     # Scale keypoints from resized frame (224x224) back to original frame size (1280x720)
#     scale_x_lower = 2000/224
#     scale_y_lower = 1100/224
#     scale_x_middle=1970/224
#     scale_y_middle=1100/224
#     scale_x_higher=1945/224
#     scale_y_higher=1100/224
#     keypoints_original=copy.deepcopy(keypoints)
#     for i in range(len(keypoints)):
#         if i in [0,2,4,5,8,10]:
#             keypoints_original[i] = keypoints[i] * np.array([scale_x_lower, scale_y_lower])
#         elif i in [12,13]:
#             keypoints_original[i] = keypoints[i] * np.array([scale_x_middle, scale_y_middle])
#         elif i in [1,3,7,9,11]:
#             keypoints_original[i] = keypoints[i] * np.array([scale_x_higher, scale_y_higher])
#     print('keypoints_original:',keypoints_original)
#     # Draw circles around keypoints on the original frame size (1280x720)
#     for point in keypoints_original:
#         print('point:',point)
#         x, y = int(point[0]), int(point[1])
#         cv2.circle(frame, (x, y), radius=5, color=(255, 0, 0), thickness=-1)  # Draw filled circle
#
#     # Write the updated frame to the output video
#     out.write(np.uint8(frame * 255.0))  # Convert frame back to uint8 before writing
#
#     # Display frame count
#     current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#     print(f'Processed frame {current_frame}/{total_frames}')
#
# # Release the VideoCapture and VideoWriter objects
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
# print(f"Video saved to {output_video_path}")
#
# from ultralytics import YOLO
# model=YOLO('yolov8x.pt')
# results=model.track(r"D:\tennis_yolo_video\input_video.mp4",conf=0.15,save=True)
# print(results)
# #results is a list
# print("=============================================")
# print("boxes:")
# for box in results[0].boxes:
#     print(box)