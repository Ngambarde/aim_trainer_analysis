if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("YOLO models/yolov8n.pt")

    model.train(
        data = "dataset/data.yaml",
        epochs = 50,
        imgsz = (1920,1080),
        batch = 16,
        patience = 10,
        dropout=0.5,
        device = "cuda"
    )