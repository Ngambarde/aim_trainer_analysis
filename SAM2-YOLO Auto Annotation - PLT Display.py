import os
import cv2
import torch
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Warning: Could not import SAM2. Ensure the 'sam2' directory is accessible.")
    SAM2ImagePredictor = None
    build_sam2 = None



# --- CONFIGURATION (Keep as is) ---
IMAGE_DIR = "images/"
OUTPUT_DIR = "outputs/"
REJECTED_DIR = os.path.join(OUTPUT_DIR, "rejected_images")
CONFIDENCE_THRESHOLD = 0.5
LABEL_LIMIT = 500 # Set to float('inf') to disable limit

YOLO_MODEL_PATH = "models/best.pt"
SAM2_CHECKPOINT = "sam2_checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "sam2_configs/sam2.1/sam2.1_hiera_l.yaml"

# --- SETUP (Keep as is) ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load models
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if build_sam2 and SAM2ImagePredictor:
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
    else:
        print("EXITING: SAM2 model could not be loaded")
        sys.exit(1)
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# --- Helper Functions (Keep generate_point_prompts, predict_sam_masks, save_yolo_segmentation, get_processed_stems as is) ---
def detect_yolo_boxes(model, image):
    """
    Runs YOLO prediction and returns boxes as a NumPy array
    """
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    boxes_tensor = results[0].boxes.xyxy
    if boxes_tensor.numel() == 0:
        return np.empty((0, 4), dtype=np.float32)
    return boxes_tensor.cpu().numpy().astype(float)

def generate_point_prompts(boxes, image_shape):
    """
    Generates point prompts, filtering out crosshair boxes
    """
    H, W, _ = image_shape
    image_middle_x = W / 2
    image_middle_y = H / 2
    point_coords_list, point_labels_list, filtered_boxes_list = [], [], []

    for x1, y1, x2, y2 in boxes:
        fg_x = (x1 + x2) / 2
        fg_y = (y1 + y2) / 2

        if abs(fg_x - image_middle_x) <= 1 and abs(fg_y - image_middle_y) <= 1:
            continue

        fg_points = [[fg_x, fg_y]]
        bg_points = [[x1, y1], [x2, y1], [x1, y2], [x2, y2], [image_middle_x, image_middle_y]]
        points = fg_points + bg_points
        labels = [1] + [0] * len(bg_points)

        point_coords_list.append(np.array(points, dtype=np.float32))
        point_labels_list.append(np.array(labels, dtype=np.int32))
        filtered_boxes_list.append([x1, y1, x2, y2])

    return point_coords_list, point_labels_list, np.array(filtered_boxes_list, dtype=np.float32)

def predict_sam_masks(predictor, image_rgb, boxes, point_coords, point_labels):
    """
    Sets image and runs SAM prediction
    """
    if len(boxes) == 0:
        return np.empty((0, image_rgb.shape[0], image_rgb.shape[1])), np.empty(0), np.empty(0)

    predictor.set_image(image_rgb)
    try:
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
            multimask_output=False
        )
        return masks, scores, logits
    except Exception as e:
        print(f"Error during SAM prediction: {e}")
        return np.empty((0, image_rgb.shape[0], image_rgb.shape[1])), np.empty(0), np.empty(0)

def save_yolo_segmentation(image_path, image_shape, masks):
    """
    Saves segmentation masks as PNGs and generates corresponding YOLOv8 segmentation labels
    """
    name_stem = Path(image_path).stem
    label_file = os.path.join(OUTPUT_DIR, "labels", f"{name_stem}.txt")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")

    # Ensure directories exist
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    H, W = image_shape[:2] # Get image height and width

    yolo_label_lines = [] # Store lines for the label file

    for i, mask in enumerate(masks):
        # Ensure mask is 2D boolean
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        mask_bool = (mask > 0)

        # Converts boolean mask to uint8 binary image for contour finding
        mask_bin = mask_bool.astype(np.uint8) * 255

        # Save the binary mask PNG
        mask_path_obj = Path(mask_dir) / f"{name_stem}_{i}.png"
        try:
            cv2.imwrite(str(mask_path_obj), mask_bin)
        except Exception as e:
            print(f"Error saving mask PNG {mask_path_obj}: {e}")
            continue # Skip this mask if saving failed

        # Find contours to get polygon points
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Warning: No contours found for mask {i} in {image_path}, skipping label entry.")
            continue

        # Assume the largest contour is the object
        contour = max(contours, key=cv2.contourArea)

        # --- Normalize contour points for YOLO format ---
        # Contour is [[x1, y1], [x2, y2], ...]
        # Normalized list: x1/W y1/H x2/W y2/H ...
        if len(contour) >= 3: # Need at least 3 points for a polygon
            normalized_points = []
            for point in contour.reshape(-1, 2): # Reshape contour to list of [x, y]
                norm_x = point[0] / W
                norm_y = point[1] / H
                normalized_points.extend([norm_x, norm_y])

            # Format the line for the label file: class_id followed by normalized points
            # Assuming class_id is 0 for 'Target'
            class_id = 0
            label_line = f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", normalized_points))
            yolo_label_lines.append(label_line)
        else:
             print(f"Warning: Contour too small for mask {i} in {image_path}, skipping label entry.")


    # --- Write all lines to the label file ---
    if yolo_label_lines:
        try:
            with open(label_file, 'w') as f:
                f.write("\n".join(yolo_label_lines) + "\n")
        except Exception as e:
            print(f"Error writing label file {label_file}: {e}")


    # --- Save original image ---
    image_save_path = os.path.join(OUTPUT_DIR, "images", f"{name_stem}.jpg")
    # Ensure the source image exists before trying to copy/save
    if Path(image_path).exists():
        try:
            source_image = cv2.imread(image_path)
            if source_image is not None:
                 cv2.imwrite(image_save_path, source_image)
            else:
                 print(f"Warning: Could not read source image {image_path} to copy.")
        except Exception as e:
             print(f"Error copying original image {image_path}: {e}")
    else:
        print(f"Warning: Source image path not found: {image_path}")


def visualize_annotation(image_rgb, masks, boxes, point_coords, point_labels):
    """
    Creates visualization using Matplotlib, showing mask count and overlay
    """
    vis_image_rgb = image_rgb.copy()

    # Calculate the number of masks generated
    num_masks = len(masks) if masks is not None and masks.size > 0 else 0

    # Draw SAM2 Masks
    overlay = np.zeros_like(vis_image_rgb, dtype=np.uint8)
    if num_masks > 0:
        for i, mask in enumerate(masks):
            if mask.ndim > 2: mask = np.squeeze(mask)
            if mask.dtype != bool: mask_bool = mask > 0.0
            else: mask_bool = mask

            if mask_bool.shape != overlay.shape[:2]:
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

            color_rgb = DISTINCT_COLORS_BGR[i % len(DISTINCT_COLORS_BGR)][::-1]
            overlay[mask_bool] = color_rgb

    # Blend overlay
    vis_image_rgb = cv2.addWeighted(vis_image_rgb, 0.6, overlay, 0.4, 0)

    # --- Display using Matplotlib ---
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.imshow(vis_image_rgb)

    # --- Create Info Text with Mask Count ---
    info_text = f"Detected: {num_masks} | [Y] Accept    [N] Reject    [E] Exit"
    ax.text(10, 30, info_text, fontsize=12,
            color='white', backgroundcolor='black', alpha=0.7)

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show(block=False)
    plt.pause(0.1)

    return fig # Return the figure object

# --- Annotation Function ---
def annotate_image(image_path, sam_predictor):
    """
    Processes a single image: detect, predict masks, visualize, takes user input.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    boxes = detect_yolo_boxes(yolo_model, image_bgr)
    if boxes.size == 0:
        return None

    point_coords, point_labels, filtered_boxes = generate_point_prompts(boxes, image_bgr.shape)
    if filtered_boxes.size == 0:
        return None

    masks, scores, _ = predict_sam_masks(sam_predictor, image_rgb, filtered_boxes, point_coords, point_labels)
    if masks.size == 0:
         return None

    # Visualize and get the figure object
    fig = visualize_annotation(image_rgb, masks, filtered_boxes, point_coords, point_labels)

    # Get user decision
    while True:
        key = input("Press 'y' to accept, 'n' to reject, 'e' to exit: ").strip().lower()
        if key in ['y', 'n', 'e']:
            plt.close(fig) # Explicitly close the Matplotlib figure, this is used to reduce memory usage and avoid crashes
            break
        else:
            print("Invalid input. Please press 'y', 'n', or 'e'.")

    # Handle decision
    if key == 'e':
        print("Exiting annotation.")
        return 'exit'
    elif key == 'n':
        rejected_path = Path(REJECTED_DIR) / Path(image_path).name
        cv2.imwrite(str(rejected_path), image_bgr)
        return False
    else: # key == 'y'
        save_yolo_segmentation(image_path, image_bgr, masks)
        return True


def get_processed_stems(output_dir, rejected_dir):
    """
    Gets a set of image stems that have already been processed from the labeled and rejected directories
    """
    processed_stems = set()
    labels_dir = Path(output_dir) / "labels"
    if labels_dir.exists():
        processed_stems.update(f.stem for f in labels_dir.glob("*.txt"))
    rejected_dir_path = Path(rejected_dir)
    if rejected_dir_path.exists():
         processed_stems.update(f.stem for f in rejected_dir_path.glob("*.jpg"))
    return processed_stems

# --- Main loop ---
def main():
    all_image_paths = list(Path(IMAGE_DIR).glob("*.jpg")) + list(Path(IMAGE_DIR).glob("*.png")) # Include png
    if not all_image_paths:
        print(f"Exiting: No images found in {IMAGE_DIR}")
        return

    processed_stems = get_processed_stems(OUTPUT_DIR, REJECTED_DIR)
    print(f"Found {len(processed_stems)} already processed images")

    images_to_process = [p for p in all_image_paths if p.stem not in processed_stems]
    if not images_to_process:
        print("All images have been processed already")
        return

    print(f"Found {len(images_to_process)} images to process")

    labeled_count = 0
    images_to_process_limited = images_to_process[:LABEL_LIMIT] if LABEL_LIMIT != float('inf') else images_to_process

    # Ensure SAM predictor is loaded
    if not sam2_predictor:
       print("Exiting: SAM2 Predictor not available")
       return

    # Process images
    for img_path in tqdm(images_to_process_limited, desc="Labeling images"):
        try:
            result = annotate_image(str(img_path), sam2_predictor)

            if result == 'exit':
                break
            elif result is True:
                labeled_count += 1

            # --- Trigger garbage collection periodically ---
            if (labeled_count + len(processed_stems)) % 20 == 0: # Runs every 20 images
                 gc.collect()
                 if DEVICE.type == 'cuda':
                     torch.cuda.empty_cache() # Clears GPU cache

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            plt.close('all')
            continue

    print(f"\nAnnotation session finished. Labeled {labeled_count} new images in this session.")
    print(f"Total labeled images (approx): {len(get_processed_stems(OUTPUT_DIR, REJECTED_DIR))}")

if __name__ == "__main__":
    DISTINCT_COLORS_BGR = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)]
    main()