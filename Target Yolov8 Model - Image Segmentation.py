import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import datetime

if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("YOLO models/yolo11n-seg.pt")

    data_yaml_path = "dataset/data.yaml"

    epochs = 100 # Epochs
    imgsz = 1920 # High-res
    batch = 6    # Batch
    optimizer = "AdamW" # Optimizer
    dropout = 0.3     # Dropout
    device = "cuda" # Use CUDA device for training
    patience = 10     # Early stopping
    name = "yolov11n_seg_v1"

    # --- Store parameters for logging ---
    training_params = {
        "model_variant": "yolov8n-seg.pt",
        "data_yaml": data_yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch_size": batch,
        "optimizer": optimizer,
        "dropout": dropout,
        "patience": patience,
        "device": device,
        "name": name,
    }

    # --- Train the Model ---
    results = model.train(
        data = data_yaml_path,
        epochs = epochs,
        imgsz = imgsz,
        batch = batch,
        optimizer = optimizer,
        dropout = dropout,
        device = device,
        patience = patience,
        name = name,
    )

    def get_dataset_size(data_yaml_path):
        """
        Parses data.yaml to count images in train, val, test sets
        """
        sizes = {'train': 0, 'val': 0, 'test': 0}
        try:
            with open(data_yaml_path, 'r') as f:
                data_cfg = yaml.safe_load(f)

            base_path = Path(data_cfg.get('path', Path(data_yaml_path).parent))

            for split in ['train', 'val', 'test']:
                split_path_key = split  # Map split name directly to key in yaml
                if split_path_key in data_cfg and data_cfg[split_path_key]:
                    # Path is relative to 'path' in yaml or yaml file's dir
                    image_dir = base_path / data_cfg[split_path_key]
                    if image_dir.is_dir():
                        # Count all images from common image types
                        count = len(list(image_dir.glob('*.jpg'))) + \
                                len(list(image_dir.glob('*.png'))) + \
                                len(list(image_dir.glob('*.jpeg'))) + \
                                len(list(image_dir.glob('*.bmp'))) + \
                                len(list(image_dir.glob('*.tif')))
                        sizes[split] = count
                    else:
                        print(f"Warning: Directory not found for split '{split}': {image_dir}")
                else:
                    print(f"Warning: Path for split '{split}' not defined or empty in {data_yaml_path}")

        except Exception as e:
            print(f"Error reading dataset size from {data_yaml_path}: {e}")
        return sizes


    def log_training_results(results, params, dataset_sizes, log_file_path):
        """
        Logs training parameters and best results to a text file
        """
        try:
            log_content = f"Training Summary ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"

            log_content += "[Training Parameters]\n"
            for key, value in params.items():
                log_content += f"- {key}: {value}\n"

            log_content += "\n[Dataset Size]\n"
            log_content += f"- Train: {dataset_sizes.get('train', 'N/A')} images\n"
            log_content += f"- Validation: {dataset_sizes.get('val', 'N/A')} images\n"
            log_content += f"- Test: {dataset_sizes.get('test', 'N/A')} images\n"

            log_content += "\n[Best Model Results]\n"
            best_epoch = results.epoch
            best_weights_path = results.save_dir / 'weights' / 'best.pt'

            # Box Metrics
            box_map50 = getattr(results.box, 'map50', 'N/A')
            box_map50_95 = getattr(results.box, 'map', 'N/A')

            # Mask Metrics
            mask_map50 = getattr(results.mask, 'map50', 'N/A')
            mask_map50_95 = getattr(results.mask, 'map', 'N/A')

            log_content += f"- Best Epoch: {best_epoch}\n"
            log_content += f"- Best Weights Saved To: {best_weights_path}\n"
            log_content += f"- Box mAP50: {box_map50:.4f}\n"
            log_content += f"- Box mAP50-95: {box_map50_95:.4f}\n"
            log_content += f"- Mask mAP50: {mask_map50:.4f}\n"
            log_content += f"- Mask mAP50-95: {mask_map50_95:.4f}\n"

            with open(log_file_path, 'w') as f:
                f.write(log_content)
            print(f"\nTraining summary saved to: {log_file_path}")

        except AttributeError as ae:
            print(f"Error logging results: Attribute missing in 'results' object - {ae}")
            print("Metrics might not be available or named differently in this Ultralytics version.")
            print("Raw results object:", results)
        except Exception as e:
            print(f"Error writing training log to {log_file_path}: {e}")