from ultralytics import YOLO
model=YOLO('D:\Tennis_yolo\models\yolo_v5_best.pt')
results=model.track(r"D:\tennis_yolo_video\input_video.mp4",conf=0.15,save=True,save_dir=r'D:\tennis_yolo_player_detection\video1')
print(results)
#results is a list
print("=============================================")
print("boxes:")
for box in results[0].boxes:
    print(box)

