from ultralytics import YOLO
import cv2


model = YOLO("models/best.pt")

# Define path to test image
image_path = "images/IMAGE_PATH.jpg"

# Run inference on the image
results = model.predict(image_path,
                        conf=0.5,
                        imgsz=1920,
                        iou = 0.4,
                        retina_masks=False
                        )

# Loop over each result and plot the segmentation masks
for result in results:
    # Adjust to display labels, masks, and bounding boxes as needed.
    segmented_img = result.plot(labels=True,
                                masks=True,
                                boxes=False
                                )

    # Display the resulting image using OpenCV
    cv2.imshow("YOLOv8 Segmentation Inference", segmented_img)

    # Wait  until a key is pressed
    cv2.waitKey(0)

    cv2.destroyAllWindows()