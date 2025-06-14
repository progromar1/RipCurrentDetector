from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # build a new model from scratch

# Train the model with 2 GPUs
results = model.train(data="path.yaml", epochs=5, batch = -1, patience = 2, device="mps")