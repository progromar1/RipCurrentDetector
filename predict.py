from ultralytics import YOLO

# Load a model
model = YOLO("/Users/omarhaq/Desktop/RiptideDetection/Code/runs/detect/train22/weights/best.pt")

# Defiend path to video source
source = "/Users/omarhaq/Desktop/with_rips/rip_05.png"
# Customize validation settings
results = model.predict(source, save_frames = True, save_txt = True, save_conf = True, save_crop = True, save = True, conf = 0.7)