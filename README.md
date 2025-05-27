# Aim Training Performance Analyzer

This project analyzes gameplay videos from aim training software (Currently limited to Kovaaks) to extract detailed performance metrics. It uses YOLOv8 for target segmentation and PaddleOCR for extracting scores and other on-screen text.

## Features:
- Target detection and segmentation using YOLOv8.
- OCR for scenario timer reset detection and results screen parsing.
- Detailed flick and adjustment phase timing.
- Calculation of metrics like time on target, time between hits, flick speed, etc.
- Batch processing for multiple video files.
- Optional debug visualization video output.
- CSV output for hit-by-hit metrics and per-video summaries.

## Setup:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Ngambarde/aim_trainer_analysis.git
    ```

2.  Create the Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Place YOLO Model:
    Ensure the YOLO model (best.pt) is correctly located in the models/ subdirectory
    You may specify a different path using the --yolo_model CLI option

## Usage
Run the script from the command line:
```bash
python aim_analysis.py path/to/your/input_videos_folder path/to/your/output_folder [OPTIONS]
```

Example: python aim_analysis.py "C:/MyAimVideos" "C:/AnalysisOutput" --no_viz

## Available Options:
--yolo_model: Path to YOLO model best.pt (or engine.pt) file.
--no_viz: Disable debug video output.
--flick_radius: Radius for flick proximity detection.
--conf_thresh: YOLO confidence threshold.
--iou_thresh: YOLO IOU threshold.
for more details run
```bash
python aim_analysis.py -h
```


## Future Goals:
- Develop GUI for easier configuration and file selection as well as metric visualizations.
- Add support for Aimlabs
- Add more advanced statstical analysis/plotting
- Add support for other scenario types
