import cv2
import numpy as np
import torch
import re
import statistics
import pandas as pd
import math
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from tqdm import tqdm
from paddleocr import PaddleOCR
import os
from pathlib import Path
import argparse
import traceback
from typing import Dict, Any, Tuple, List, Optional

# --- CONFIGURATION ---
# Default configuration, can be overridden by CLI args or in batch_process_folder with cli_config_overrides
DEFAULT_APP_CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "YOLO_MODEL_PATH": Path("models/best.pt"),
    "YOLO_CONF_THRESH": 0.5,    # Labels any object detected with >50% confidence
    "YOLO_IOU_THRESH": 0.4,     # 0.4 ensure targets are not double counted
    "PROXIMITY_RADIUS": 512,    # Square range around crosshair that YOLO predicts within (512 = 1024x1024)
    "CROSSHAIR_PAD": 6,         # Essentially makes the crosshair an X pixel diameter circle instead of one point
    "TARGET_MOTION_MAX_DIST": 100,  # Only grabs masks within 100px radius of previous masks in Hungarian algorithm
    "PREV_TARGET_TRACKING_THRESH": 15,  # Range in which the previous targets center is compared, larger = false positives, smaller = false negatives
    "FLICK_PROXIMITY_RADIUS": 50,   # X pixel diameter sphere on target centers, used to determine when the crosshair is in "adjustment" range (as well as using speed threshold)
    "FLICK_MIN_SPEED_START_THRESHOLD": 15, # Flicks do not start until crosshair moves at >X px
    "FLICK_MAX_SPEED_END_THRESHOLD": 15,    # Flick ends when crosshair speed <X
    "OCR_FRAME_SCAN_PERCENTAGE_END": 0.95,  # OCR scans the last x% of frames for the results screen, could make this more robust in the future
    "OCR_TIMER_RESET_REGEX_PATTERN": r'\b0?:59\b',  # Stores pattern (default = r'\b0?:59\b') adjustable in case scenarios are not 60 seconds
    "OCR_INITIAL_SCAN_DURATION_PCT": 0.30,  # OCR scans the first x% of frames for 0:59 (indicating a scenario reset)
    "OCR_TIMER_PADDING": 8, # Padding around OCR timer bounding box
    "ENABLE_DEBUG_VISUALIZATION": True, # True = output video with debug, False = No video output
}

DEFAULT_APP_CONFIG["OCR_TIMER_RESET_REGEX"] = re.compile(DEFAULT_APP_CONFIG["OCR_TIMER_RESET_REGEX_PATTERN"]) # Compiles regex for timer


# --- Helper classes --- #
class AimPhaseTracker:
    """
    This class/function manages the state and timing for the flick and adjustment phases
    """
    IDLE = 0
    WAITING_FLICK_SPEED = 1
    TIMING_FLICK = 2
    TIMING_ADJUSTMENT = 3

    def __init__(self, config: Dict[str, Any], fps: float):
        self.config = config
        self.fps = fps
        if self.fps == 0:  # Safety for FPS
            print("Warning: AimPhaseTracker initialized with FPS=0. Setting default to 30 for calculations.")
            self.fps = 30.0

        self.current_phase = self.IDLE          # Sets initial object state to IDLE
        self.frames_in_current_phase = 0        # Sets current frames in phase to 0
        self.accumulated_flick_distance = 0.0   # Sets initial flick distance to 0 px
        self.last_flick_metrics = {}            # Stores the metrics of the last flick is processed

    def reset_after_hit(self):
        """
        Resets state and waits for next flick after a hit
        """
        self.current_phase = self.WAITING_FLICK_SPEED       # Sets object phase to waiting flick speed
        self.frames_in_current_phase = 0
        self.accumulated_flick_distance = 0.0
        self.last_flick_metrics = {}                        # Clear last flick metrics for the new cycle

    def update(self, current_crosshair_speed: float, in_flick_proximity: bool, hit_registered_this_frame: bool) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Updates the aiming phase and timings
        """
        flick_metrics_output = None
        adjustment_metrics_output = None

        # --- State Transitions & Logic ---
        if self.current_phase == self.WAITING_FLICK_SPEED:
            self.frames_in_current_phase += 1
            if current_crosshair_speed >= self.config["FLICK_MIN_SPEED_START_THRESHOLD"]:
                self.current_phase = self.TIMING_FLICK
                self.frames_in_current_phase = 1  # Start counting frames for flick
                self.accumulated_flick_distance = current_crosshair_speed  # Include speed of first frame
            # If a hit occurs while waiting for flick speed, reset
            elif hit_registered_this_frame:
                self.reset_after_hit()


        elif self.current_phase == self.TIMING_FLICK:
            self.frames_in_current_phase += 1
            self.accumulated_flick_distance += current_crosshair_speed

            flick_ended = (in_flick_proximity and
                           current_crosshair_speed < self.config["FLICK_MAX_SPEED_END_THRESHOLD"])

            if hit_registered_this_frame:  # Hit during flick (can happen crosshair speed never drops below threshold)
                flick_metrics_output = self._calculate_flick_metrics(current_crosshair_speed)
                # Essentially zero adjustment time
                adjustment_metrics_output = {"adjustment_time_s": round(1 / self.fps, 4)}
                self.reset_after_hit()
            elif flick_ended:  # Flick ends, transition to adjustment
                flick_metrics_output = self._calculate_flick_metrics(current_crosshair_speed)
                self.last_flick_metrics = flick_metrics_output.copy()  # Store for when adjustment ends
                self.current_phase = self.TIMING_ADJUSTMENT
                self.frames_in_current_phase = 1  # Start counting adjustment frames
                self.accumulated_flick_distance = 0.0  # Resets for future use

        elif self.current_phase == self.TIMING_ADJUSTMENT:
            if not hit_registered_this_frame:
                self.frames_in_current_phase += 1
            else:  # Hit registered, adjustment ends
                adjustment_time_s = round(self.frames_in_current_phase / self.fps, 4)
                adjustment_metrics_output = {"adjustment_time_s": adjustment_time_s}
                # The flick metrics were calculated when the flick ended
                # This uses the stored self.last_flick_metrics if available
                flick_metrics_output = self.last_flick_metrics if self.last_flick_metrics else self._get_nan_flick_metrics()
                self.reset_after_hit()

        if hit_registered_this_frame and self.current_phase != self.WAITING_FLICK_SPEED:  # Resets if a hit happens outside the normal flow
            if not (flick_metrics_output or adjustment_metrics_output):  # If hit didn't naturally end a phase
                # This will handle cases were a hit occurred in an unexpected phase
                # Inputs NaN metrics for this hit event if no flick/adj was timed.
                flick_metrics_output = self._get_nan_flick_metrics()
                adjustment_metrics_output = {"adjustment_time_s": float('nan')}
            self.reset_after_hit()

        return flick_metrics_output, adjustment_metrics_output

    def _calculate_flick_metrics(self, speed_at_flick_end: float) -> Dict:
        """
        Calculates metrics for flick cycle when a flick ends
        """
        flick_time_s = round(self.frames_in_current_phase / self.fps, 4)
        flick_distance_px = self.accumulated_flick_distance
        norm_flick_time = float('nan')
        avg_flick_speed = float('nan')

        if flick_distance_px > 0 and flick_time_s > 0:
            norm_flick_time = round(flick_time_s / flick_distance_px, 6)    # Normalizes flick time based on distance traveled in s/px
            avg_flick_speed = round(flick_distance_px / flick_time_s, 2)    # Non-normalized, simply avg flick speed in px/s

        return {
            "flick_time_s": flick_time_s,
            "speed_at_flick_end_px_f": round(speed_at_flick_end, 2),
            "flick_distance_px": round(flick_distance_px, 2),
            "norm_flick_time_s_per_px": norm_flick_time,
            "avg_flick_speed_px_s": avg_flick_speed,
        }

    def _get_nan_flick_metrics(self) -> Dict:
        """
        Returns a dict with NaN values for all flick metrics
        """
        return {
            "flick_time_s": float('nan'),
            "speed_at_flick_end_px_f": float('nan'),
            "flick_distance_px": float('nan'),
            "norm_flick_time_s_per_px": float('nan'),
            "avg_flick_speed_px_s": float('nan'),
        }

    def get_status_text(self) -> str:
        """
        Returns a string describing the current tracking phase for visualization
        """
        if self.current_phase == self.IDLE: return "Idle"
        if self.current_phase == self.WAITING_FLICK_SPEED: return f"Wait Flick Speed (F:{self.frames_in_current_phase})"
        if self.current_phase == self.TIMING_FLICK: return f"Flicking (F:{self.frames_in_current_phase})"
        if self.current_phase == self.TIMING_ADJUSTMENT: return f"Adjusting (F:{self.frames_in_current_phase})"
        return "Unknown"


# --- Helper functions ---
def initialize_video_io(video_path: Path, out_path: Path, config: Dict) -> Tuple[
    cv2.VideoCapture, Optional[cv2.VideoWriter], float, int, int, int]:
    """
    Initializes video capture and writer
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if config["ENABLE_DEBUG_VISUALIZATION"]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 30.0, (width, height))
    return cap, writer, fps, width, height, total_frames


def load_models(yolo_model_path: Path, device: torch.device, use_gpu_ocr: bool) -> Tuple[YOLO, PaddleOCR]:
    """
    Loads YOLO and PaddleOCR models
    """
    if not yolo_model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {yolo_model_path}. Please ensure it's in the 'models' directory or provide a different path in config or via --yolo_model.")
    yolo_model = YOLO(yolo_model_path).to(device)
    ocr_model = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_gpu_ocr, show_log=False)
    return yolo_model, ocr_model


def extract_text_from_ocr_region(ocr_engine: PaddleOCR, image_region: np.ndarray) -> str:
    """
    Extracts all text from a given image region using OCR
    """
    ocr_results = ocr_engine.ocr(image_region, cls=False)
    extracted_text = ""
    if ocr_results and ocr_results[0]:  # PaddleOCR structure
        for line_info in ocr_results[0]:
            text = line_info[1][0]
            extracted_text += text + " "
    return extracted_text.strip()


def get_box_from_ocr_results(ocr_results: List, target_text_regex: re.Pattern) -> Optional[List]:
    """
    Finds the bounding box of the text matching OCR_TIMER_RESET_REGEX_PATTERN (default 0:59)
    """
    if ocr_results and ocr_results[0]:  # New PaddleOCR structure
        for line_info in ocr_results[0]:    # Searches through all detected text
            box_coords = line_info[0]
            text = line_info[1][0]
            if target_text_regex.search(text) and text.count(':') == 1: # target_text_regex = OCR_TIMER_RESET_REGEX_PATTERN
                return box_coords
    return None


def calculate_target_motion(current_centers: List, prev_centers: List, max_dist: int) -> Tuple[int, int]:
    """
    Calculates target motion, uses Hungarian algorithm for matching targets based on previous frame location
    """

    """
    cost_matrix = np.linalg.norm(cur_np[:, None, :] - prev_np[None, :, :], axis=2) explanation:

    cur_np[:, None, :] - prev_np[None, :, :] creates an (N, M, 2) array with each element containing [i, j, :]
    containing [dx, dy] which is the difference between the current and previous target centers
    np.linalg.norm calculates the euclidian distance of the values defined in the array against the last axis (axis=2)
    meaning for each [i, j] it calculate sqrt(dx^2 + dy^2)

    IN SHORT cost_matrix is a numpy array = [[Euclidian(c0,p0), Euclidian(c1,p0)],
     	                                    [Euclidian(c0,p1), Euclidian(c1,p1)],
     	                                    [Euclidian(c0,p2), Euclidian(c1,p2)]]

    row_ind, col_ind = linear_sum_assignment(cost_matrix) explanation:
    lienar_sum_assignment(cost_matrix): Solves the cost matrix using the hungarian algorithm, returns the minimum of
    current/previous targets. Meaning, if there were 3 previous and 2 current, it would return 2 coordinates and vice versa
    row_ind, col_ind: Assigning the least "costly" row and col indices to the relevant variables
    """
    if not current_centers or not prev_centers: return 0, 0          # Simply detects if either list is empty ie. no targets to compare against
    cur_np = np.array(current_centers, dtype=float).reshape(-1, 2)   # Converts list to np.array and reshapes to X rows (-1 in numpy does this) and 2 columns
    prev_np = np.array(prev_centers, dtype=float).reshape(-1, 2)     # Same as above

    cost_matrix = np.linalg.norm(cur_np[:, None, :] - prev_np[None, :, :], axis=2)  # See explanation above for more information

    row_ind, col_ind = linear_sum_assignment(cost_matrix)   # See explanation above for more information

    matched_distances = cost_matrix[row_ind, col_ind]   # Grabs the euclidian distances found in the cost matrix at the least "costly" locations
    valid_matches = matched_distances <= max_dist   # If euclidian distance <= TARGET_MOTION_MAX_DIST pass it into valid_matches, this is just a failsafe to avoid any erroneous readings, can potentially remove

    if np.any(valid_matches):   # If any valid matches exist
        good_rows, good_cols = row_ind[valid_matches], col_ind[valid_matches] # good_rows = valid current frame targets, good_cols = valid previous frame targets
        dxs = cur_np[good_rows, 0] - prev_np[good_cols, 0]      # Calculates the VECTOR (direction + magnitude) from current to previous targets x's
        dys = cur_np[good_rows, 1] - prev_np[good_cols, 1]      # Calculates the VECTOR (direction + magnitude) from current to previous targets y's
        return int(dxs.mean()) if dxs.size > 0 else 0, int(dys.mean()) if dys.size > 0 else 0   # Returns the mean x and y values, this will be the distance the crosshair moved, ie. current crosshair position + (x,y) = previous crosshair positon
    return 0, 0


def find_closest_point(points_list: List, target_point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Finds the closest target center to current crosshair position
    """

    """
    This functions is used to determine the closest target center to the crosshair.
    This is used later to help identify if a target was successfully hit, or simply passed over
    By adding the crosshair movement to its current location, if this lands near a target center, it means the target was not hit.
    If there is no target center in proximity, it confirms a hit
    """

    if not points_list: return None
    min_dist_sq = float('inf')
    closest_pt = None
    for pt_tuple in points_list:
        pt = np.array(pt_tuple)
        target = np.array(target_point)         # Converts the target center points into a numpy array
        dist_sq = np.sum((pt - target) ** 2)    # dist_sq = euclidian distance from the crosshair to targets
        if dist_sq < min_dist_sq:               # finds the closest point and assigns it to closest_pt
            min_dist_sq = dist_sq
            closest_pt = tuple(pt_tuple)        # Return in original format
    return closest_pt


def is_point_in_circle(point: Tuple[int, int], circle_center: Tuple[int, int], radius: int) -> bool:
    """
    Checks if a point is within a given circle proximity
    """
    return (point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2 < radius ** 2


def process_yolo_segmentation(frame_crop: np.ndarray, yolo_model: YOLO, config: Dict,
                              crop_offset_x: int, crop_offset_y: int,
                              frame_h: int, frame_w: int) -> Tuple[
    List[List[int]], np.ndarray, List[np.ndarray]]:  # List[np.ndarray] for segmentation mask polygons
    """
    Performs YOLO segmentation and returns mask centers, combined mask, and segmentation mask polygons
    """
    yolo_result = yolo_model.predict(source=frame_crop,
                                          imgsz=config["PROXIMITY_RADIUS"] * 2,
                                          device=config["DEVICE"],
                                          conf=config["YOLO_CONF_THRESH"],
                                          iou=config["YOLO_IOU_THRESH"],
                                          retina_masks=True, verbose=False  # Retina_masks = True allow for higher quality segmentation masks, False will reduce metric accuracy but will run faster, leave this on.
                                         )
    mask_polygons_relative = yolo_result[0].masks.xy if yolo_result[0].masks is not None else []

    current_mask_centers = []
    full_frame_combined_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    absolute_polygons_for_viz = []  # List used to store absolute polygons

    for poly_relative in mask_polygons_relative:
        poly_absolute = (np.array(poly_relative, dtype=np.int32) + np.array([crop_offset_x, crop_offset_y]))
        absolute_polygons_for_viz.append(poly_absolute)  # Stored for returning
        cv2.fillPoly(full_frame_combined_mask, [poly_absolute], 255)    # Draws the segmentation masks on the frame

        M = cv2.moments(poly_absolute)
        if M["m00"] != 0:   # m00 is equal to the area of the polygon for the mask.
            mask_center_x = int(M["m10"] / M["m00"])    # Calculates the centroid x of the polygon mask
            mask_center_y = int(M["m01"] / M["m00"])    # Calculates the centroid y of the polygon mask
            current_mask_centers.append([mask_center_x, mask_center_y])     # Appends the [x, y] coords to current_mask_centers
        else:   # Failsafe for determining target centers
            mask_x0, mask_y0, mask_w, mask_h_ = cv2.boundingRect(poly_absolute)
            current_mask_centers.append([mask_x0 + mask_w // 2, mask_y0 + mask_h_ // 2])

    return current_mask_centers, full_frame_combined_mask, absolute_polygons_for_viz  # Return mask centers, combined mask, and mask polygons


def check_scenario_reset_ocr(frame: np.ndarray, ocr_model: PaddleOCR, ocr_state: Dict, config: Dict, frame_w: int,
                             frame_h: int) -> bool:
    """
    Checks for scenario reset using OCR on the timer
    """
    scenario_reset_flag = False
    tx0, ty0, tx1, ty1 = ocr_state["timer_ocr_roi_coords"]  # Searches the ocr_state dictionary for timer coords, using full frame significantly increases processing time
    timer_scan_img = frame[ty0:ty1, tx0:tx1]                # Generates an image from the current frame from the dimensions above
    ocr_raw_output = ocr_model.ocr(timer_scan_img, cls=False)   # Runs OCR check on the timer_scan_image
    timer_box_coords_relative = get_box_from_ocr_results(ocr_raw_output, config["OCR_TIMER_RESET_REGEX"])

    if timer_box_coords_relative:
        if not ocr_state["timer_found"]:
            xs = [pt[0] for pt in timer_box_coords_relative]
            ys = [pt[1] for pt in timer_box_coords_relative]
            abs_box_x0, abs_box_y0 = tx0 + min(xs), ty0 + min(ys)
            abs_box_x1, abs_box_y1 = tx0 + max(xs), ty0 + max(ys)
            pad = config["OCR_TIMER_PADDING"]
            ocr_state["timer_ocr_roi_coords"] = [
                max(0, int(abs_box_x0 - pad)), max(0, int(abs_box_y0 - pad)),
                min(frame_w, int(abs_box_x1 + pad)), min(frame_h, int(abs_box_y1 + pad))    # Sets new ROI for subsequent OCR scans
            ]
            ocr_state["timer_found"] = True
        if not ocr_state["scenario_reset_prev_frame"]:  # Used to avoid resetting the entirety of 0:59
            scenario_reset_flag = True
    ocr_state["scenario_reset_prev_frame"] = scenario_reset_flag  # Update for next frame
    return scenario_reset_flag


def check_results_screen_ocr(frame: np.ndarray, ocr_model: PaddleOCR, game_results_ocr: Dict, config: Dict,
                             frame_w: int, frame_h: int):
    """
    Checks and extracts data from the results screen
    """
    if any(v is None for v in game_results_ocr.values()):
        if game_results_ocr["final_score"] is None:     # If score is already found this check is skipped
            score_roi = frame[:frame_h // 2, :frame_w // 2]  # Searches the top left of the screen for final score
            text_left = extract_text_from_ocr_region(ocr_model, score_roi)
            score_match = re.search(r'final\s*score[:\s]+([0-9,.]+)', text_left, re.IGNORECASE)
            if score_match: game_results_ocr["final_score"] = score_match.group(1).replace(',', '')

        center_roi_y0, center_roi_y1 = int(frame_h * 0.3), int(frame_h * 0.7)       # accuracy and hit count appear in the center area of the screen so no need to scan the entire frame for this text
        center_roi_x0, center_roi_x1 = int(frame_w * 0.3), int(frame_w * 0.7)
        center_scan_roi = frame[center_roi_y0:center_roi_y1, center_roi_x0:center_roi_x1]
        text_center = extract_text_from_ocr_region(ocr_model, center_scan_roi)

        if game_results_ocr["accuracy"] is None:        # If accuracy is already found this check is skipped
            acc_match = re.search(r'(\d+\.\d+)%', text_center)
            if acc_match: game_results_ocr["accuracy"] = acc_match.group(1) + "%"

        if game_results_ocr["hit_count"] is None:       # If hit_count is already found this check is skipped
            hits_shots_match = re.search(r'(\d+)\s*/\s*(\d+)', text_center)  # Format is Hits / Shots
            if hits_shots_match:
                game_results_ocr["hit_count"] = int(hits_shots_match.group(1))  # Stores hit count in game_results_ocr dictionary
                game_results_ocr["shot_count"] = int(hits_shots_match.group(2)) # Stores shot count in game_results_ocr dictionary


def determine_hit_registration(
        is_on_target_currently: bool,
        prev_on_target_flag: bool,
        current_mask_centers: List,
        prev_mask_centers: List,
        avg_motion_x: int, avg_motion_y: int,
        screen_center_x: int, screen_center_y: int,
        config: Dict
) -> bool:
    """
    Determines if a hit was registered in the current frame
    """
    is_previous_main_target_identifiable = False
    if current_mask_centers and prev_mask_centers:
        moved_crosshair_pos = (screen_center_x + avg_motion_x, screen_center_y + avg_motion_y)
        closest_target_to_current_crosshair = find_closest_point(current_mask_centers, moved_crosshair_pos) # Finds the target center closest to current crosshair
        prev_closest_target_to_crosshair = find_closest_point(prev_mask_centers, (screen_center_x, screen_center_y))    # Finds previous closest target center to previous crosshair locations.

        if closest_target_to_current_crosshair and prev_closest_target_to_crosshair:
            est_prev_pos_x = closest_target_to_current_crosshair[0] - avg_motion_x  # Estimates where the current closest target was on the previous frame given the crosshairs movement
            est_prev_pos_y = closest_target_to_current_crosshair[1] - avg_motion_y  # Estimates where the current closest target was on the previous frame given the crosshairs movement
            dist_x = abs(est_prev_pos_x - prev_closest_target_to_crosshair[0])      # Subtracts the estimated current closest target position, by the previous closes targets actual position
            dist_y = abs(est_prev_pos_y - prev_closest_target_to_crosshair[1])      # Subtracts the estimated current closest target position, by the previous closes targets actual position
            if (dist_x < config["PREV_TARGET_TRACKING_THRESH"] and                  # If both of these values are < PREV_TARGET_TRACKING_THRESH, it is the same target, ie. the target was missed
                    dist_y < config["PREV_TARGET_TRACKING_THRESH"]):                # If the target is not present, it means it is a different target and the target was hit.
                is_previous_main_target_identifiable = True

    return prev_on_target_flag and not is_on_target_currently and not is_previous_main_target_identifiable


def draw_debug_visualizations(frame: np.ndarray, viz_data: Dict, config: Dict):
    """
    Draws all debug visualizations onto the frame   #
    """

    # --- Draw ROIs ---
    # Timer ROI
    cv2.rectangle(frame, (viz_data["timer_roi"][0], viz_data["timer_roi"][1]),
                  (viz_data["timer_roi"][2], viz_data["timer_roi"][3]), (255, 100, 0), 2)

    # YOLO crop ROI
    cv2.rectangle(frame, (viz_data["prox_crop_coords"][0], viz_data["prox_crop_coords"][1]),
                  (viz_data["prox_crop_coords"][2], viz_data["prox_crop_coords"][3]), (255, 100, 0), 2)

    # Crosshair + padding
    cv2.circle(frame, (viz_data["screen_center_x"], viz_data["screen_center_y"]), config["CROSSHAIR_PAD"], (0, 0, 255),
               1)

    # Flick proximity radius around center
    cv2.circle(frame, (viz_data["screen_center_x"], viz_data["screen_center_y"]), config["FLICK_PROXIMITY_RADIUS"],
               (255, 255, 0), 1)  # Flick radius

    # Setting values to int for previous crosshair position location
    prev_ch_x = int(viz_data["prev_crosshair_x"])
    prev_ch_y = int(viz_data["prev_crosshair_y"])
    curr_ch_x = int(viz_data["screen_center_x"])
    curr_ch_y = int(viz_data["screen_center_y"])

    if prev_ch_x != curr_ch_x or prev_ch_y != curr_ch_y:    # Checks if any movement was detected
        arrow_color = (255, 0, 255)
        arrow_thickness = 1
        cv2.arrowedLine(frame, (prev_ch_x, prev_ch_y), (curr_ch_x, curr_ch_y),
                        arrow_color, arrow_thickness, tipLength=0.1)

    # Draws detected masks and centers
    for poly_abs in viz_data.get("mask_polygons_absolute", []):
        cv2.polylines(frame, [poly_abs], True, (0, 255, 0), 1)
    for center_x, center_y in viz_data.get("current_mask_centers", []):
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

    # --- Draw text ---
    # Draw text info
    text_y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color_info = (0, 255, 0)
    color_status = (255, 255, 0)
    color_metric = (0, 255, 255)
    color_current_metric = (255,165,0)
    thickness = 1

    # Hits
    cv2.putText(frame, f"Hits: {viz_data['current_hits']}", (10, text_y_offset), font, font_scale, color_info,
                thickness)
    text_y_offset += 18

    # Crosshair Speed
    cv2.putText(frame, f"Crosshair Speed: {viz_data['current_crosshair_speed']:.2f} px/f", (10, text_y_offset), font,
                font_scale, color_status, thickness)
    text_y_offset += 18

    # Aim Phase Status
    cv2.putText(frame, viz_data['aim_phase_status'], (10, text_y_offset), font, font_scale, color_status, thickness)
    text_y_offset += 18

    # Current Cycle Metrics
    cv2.putText(frame, f"Current ToT: {viz_data.get('current_time_on_target_s', 0.0):.3f}s", (10, text_y_offset), font, font_scale,
                color_current_metric, thickness)
    text_y_offset += 18
    cv2.putText(frame, f"Current TBH: {viz_data.get('current_time_between_hits_s', 0.0):.3f}s", (10, text_y_offset), font, font_scale,
                color_current_metric, thickness)
    text_y_offset += 18

    # Last Completed Hit Metrics
    text_y_offset += 5 # Adding a small gap between last hit metrics

    # Last time on target (seconds)
    if viz_data['last_tot_s'] is not None:
        cv2.putText(frame, f"Last ToT: {viz_data['last_tot_s']:.3f}s", (10, text_y_offset), font, font_scale,
                    color_metric, thickness)
        text_y_offset += 20

    # Last flick/adjust if both exist for cleaner display
    last_flick_text = "Last Flick: N/A"
    if viz_data['last_flick_s'] is not None and not np.isnan(viz_data['last_flick_s']):
        flick_s = viz_data['last_flick_s']
        flick_speed = viz_data.get('last_flick_speed_pxf', float('nan'))
        if not np.isnan(flick_speed):
            last_flick_text = f"Last Flick: {flick_s:.3f}s @ {flick_speed:.1f}px/f"
        else:
            last_flick_text = f"Last Flick: {flick_s:.3f}s"
    cv2.putText(frame, last_flick_text, (10, text_y_offset), font, font_scale, color_metric, thickness)
    text_y_offset += 18

    last_adjust_text = "Last Adjust: N/A"
    if viz_data['last_adj_s'] is not None and not np.isnan(viz_data['last_adj_s']):
        last_adjust_text = f"Last Adjust: {viz_data['last_adj_s']:.3f}s"
    cv2.putText(frame, last_adjust_text, (10, text_y_offset), font, font_scale, color_metric, thickness)
    text_y_offset += 18


# --------------- MAIN PROCESSING LOGIC ---------------
def main_process_video(
        video_file_path: Path,      # Path to input video
        output_video_path: Path,    # Path to output video (if debug is enabled)
        output_csv_dir: Path,       # Path to output directory
        config: Dict[str, Any]      # Path to config file
):
    """
    Main loop for processing the aim training video
    """

    print(f"Initializing analysis for: {video_file_path.name}")
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    cap, writer, fps, frame_w, frame_h, total_frames = initialize_video_io(
        video_file_path, output_video_path, config
    )
    if fps == 0:
        print(f"Warning: FPS for {video_file_path.name} is 0. Defaulting to 30 FPS.")
        fps = 30.0

    yolo_model, ocr_model = load_models(config["YOLO_MODEL_PATH"], config["DEVICE"], torch.cuda.is_available())

    # --- Screen & ROI Coordinates ---
    screen_center_x, screen_center_y = frame_w // 2, frame_h // 2
    prox_radius = config["PROXIMITY_RADIUS"]
    prox_crop_y0, prox_crop_y1 = max(screen_center_y - prox_radius, 0), min(screen_center_y + prox_radius, frame_h)
    prox_crop_x0, prox_crop_x1 = max(screen_center_x - prox_radius, 0), min(screen_center_x + prox_radius, frame_w)
    ch_pad = config["CROSSHAIR_PAD"]
    crosshair_roi_y0, crosshair_roi_y1 = max(0, screen_center_y - ch_pad), min(frame_h, screen_center_y + ch_pad)
    crosshair_roi_x0, crosshair_roi_x1 = max(0, screen_center_x - ch_pad), min(frame_w, screen_center_x + ch_pad)

    # --- State Variables ---
    hit_events_data = []  # Stores dictionary of metrics for each hit

    # Frame-level metrics accumulation
    frames_on_target_current_cycle = 0
    frames_between_hits_current_cycle = 0

    # Inter-frame states for core logic
    prev_mask_centers = []
    prev_on_target_flag = False

    # OCR states
    ocr_state = {
        "timer_ocr_roi_coords": [0, 0, frame_w, frame_h],  # Initial full frame scan for timer
        "timer_found": False,
        "scenario_reset_prev_frame": False,  # To avoid multiple resets from multiple checks where timer = 0:59
    }
    game_results_ocr = {"final_score": None, "accuracy": None, "hit_count": None, "shot_count": None}

    # Aim Phase Tracking
    aim_phase_tracker = AimPhaseTracker(config, fps)
    aim_phase_tracker.reset_after_hit()  # Starts at WAITING_FOR_FLICK_SPEED

    # --- Main Loop ---
    for frame_idx in tqdm(range(total_frames), desc=f"Processing {video_file_path.name}", unit="frame"):    # Runs loops until all frames are exhausted
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: Could not read frame {frame_idx} from {video_file_path.name} or video ended.")
            break

        # 1. OCR scenario reset timer check
        if frame_idx < config["OCR_INITIAL_SCAN_DURATION_PCT"] * total_frames and (frame_idx % 10) == 0: # Checks every 10 frames for the first x% of the video (x% is set from OCR_INITIAL_SCAN_DURATION_PCT)
            if check_scenario_reset_ocr(frame, ocr_model, ocr_state, config, frame_w, frame_h):          # Checks if timer = 0:59
                print(f"Scenario reset detected at frame {frame_idx} in {video_file_path.name}")
                hit_events_data = []  # Reset all collected hit data
                frames_on_target_current_cycle = 0
                frames_between_hits_current_cycle = 0
                aim_phase_tracker.reset_after_hit()
                game_results_ocr = {"final_score": None, "accuracy": None, "hit_count": None, "shot_count": None} # Resets all game_results_ocr values

        # 2. OCR results screen scan
        if frame_idx > int(config["OCR_FRAME_SCAN_PERCENTAGE_END"] * total_frames): # Checks every frame for the final x% of the video for the results screen (x% set with OCR_FRAME_SCAN_PERCENTAGE_END)
            check_results_screen_ocr(frame, ocr_model, game_results_ocr, config, frame_w, frame_h)

        # 3. YOLO target segmentation
        yolo_input_crop = frame[prox_crop_y0:prox_crop_y1, prox_crop_x0:prox_crop_x1]   # Runs YOLO prediction on the cropped frame region
        current_mask_centers, full_frame_combined_mask, mask_polygons_for_drawing = process_yolo_segmentation(  # Captures the mask information obtained from the prediction
            yolo_input_crop, yolo_model, config, prox_crop_x0, prox_crop_y0, frame_h, frame_w
        )

        # 4. Target motion estimation
        avg_motion_x, avg_motion_y = calculate_target_motion(
            current_mask_centers, prev_mask_centers, config["TARGET_MOTION_MAX_DIST"]
        )
        current_crosshair_speed = math.sqrt(avg_motion_x ** 2 + avg_motion_y ** 2)

        # 5. On-target check
        is_on_target_currently = bool(full_frame_combined_mask[
                                      crosshair_roi_y0:crosshair_roi_y1, crosshair_roi_x0:crosshair_roi_x1
                                      ].any())  # Checks if the crosshair + padding falls within a target masks, if so returns True

        if is_on_target_currently:
            frames_on_target_current_cycle += 1 # Counts frames on target when above is true

        # 6. Hit registration
        hit_registered_this_frame = determine_hit_registration(
            is_on_target_currently, prev_on_target_flag, current_mask_centers, prev_mask_centers,
            avg_motion_x, avg_motion_y, screen_center_x, screen_center_y, config
        )   # Calls the determine_hit_registration function, see function for more details

        # 7. Update aim phase tracker
        in_flick_proximity_of_a_target = False
        if current_mask_centers:
            for target_center in current_mask_centers:
                if is_point_in_circle((screen_center_x, screen_center_y), target_center,
                                      config["FLICK_PROXIMITY_RADIUS"]):    # Determines if crosshair is currently in FLICK_PROXIMITY_RADIUS of current targets
                    in_flick_proximity_of_a_target = True
                    break

        flick_metrics, adjustment_metrics = aim_phase_tracker.update(
            current_crosshair_speed, in_flick_proximity_of_a_target, hit_registered_this_frame
        )

        # 8. Metric collection on hit
        if hit_registered_this_frame:
            current_hit_data = {    # Grabs all the accumulated metric information
                "hit_number": len(hit_events_data) + 1,
                "time_on_target_s": round(frames_on_target_current_cycle / fps, 4),
                "time_between_hits_s": round(frames_between_hits_current_cycle / fps, 4),
            }
            current_hit_data.update(flick_metrics if flick_metrics else aim_phase_tracker._get_nan_flick_metrics())
            current_hit_data.update(adjustment_metrics if adjustment_metrics else {"adjustment_time_s": float('nan')})

            hit_events_data.append(current_hit_data)

            # Reset cycle counters for next hit
            frames_on_target_current_cycle = 0
            frames_between_hits_current_cycle = 0
            # aim_phase_tracker reset_after_hit is called internally and not reset here

        frames_between_hits_current_cycle += 1  # Increments every frame, is reset when hit is registered

        # 9. Prepare for next frame
        prev_mask_centers = current_mask_centers.copy() # Stores current mask centers in previous mask centers
        prev_on_target_flag = is_on_target_currently    # Stores is on target to prev on target

        # 10. Visualization (if enabled)
        if config["ENABLE_DEBUG_VISUALIZATION"] and writer is not None:
            # Collecting data for visualization
            prev_crosshair_x = screen_center_x - avg_motion_x
            prev_crosshair_y = screen_center_y - avg_motion_y
            viz_data = {
                "timer_roi": ocr_state["timer_ocr_roi_coords"],
                "prox_crop_coords": (prox_crop_x0, prox_crop_y0, prox_crop_x1, prox_crop_y1),
                "screen_center_x": screen_center_x,
                "screen_center_y": screen_center_y,
                "effective_prev_crosshair_x": prev_crosshair_x,
                "effective_prev_crosshair_y": prev_crosshair_y,
                "current_hits": len(hit_events_data),
                "current_crosshair_speed": current_crosshair_speed,
                "aim_phase_status": aim_phase_tracker.get_status_text(),

                # Current cycle metrics
                "current_time_on_target_s": frames_on_target_current_cycle / fps if fps > 0 else 0,
                "current_time_between_hits_s": frames_between_hits_current_cycle / fps if fps > 0 else 0,

                # Last completed hit metrics
                "last_tot_s": hit_events_data[-1]["time_on_target_s"] if hit_events_data else None,
                "last_flick_s": hit_events_data[-1]["flick_time_s"] if hit_events_data else None,
                "last_flick_speed_pxf": hit_events_data[-1]["speed_at_flick_end_px_f"] if hit_events_data else None,
                "last_adj_s": hit_events_data[-1]["adjustment_time_s"] if hit_events_data else None,

                # Absolute polygon coordinates for drawing
                "mask_polygons_absolute": mask_polygons_for_drawing, # Use the variable passed from process_yolo_segmentation
                "current_mask_centers": current_mask_centers
            }
            draw_debug_visualizations(frame, viz_data, config)
            writer.write(frame)

    # --- Post-processing and saving results ---
    cap.release()
    if writer is not None and writer.isOpened(): writer.release()
    cv2.destroyAllWindows()

    # Creating DataFrames from collected data
    df_hits = pd.DataFrame()
    if hit_events_data:
        df_hits = pd.DataFrame(hit_events_data)
        # Correct hit_count from OCR if it differs from detected hits
        if game_results_ocr.get("hit_count") is not None:
            ocr_hit_count = int(game_results_ocr["hit_count"])
            if ocr_hit_count != len(df_hits) and abs(ocr_hit_count - len(df_hits)) <= 2:
                print(f"Adjusting hit events based on OCR results: {len(df_hits)} -> {ocr_hit_count}")
                if ocr_hit_count < len(df_hits):
                    df_hits = df_hits.head(ocr_hit_count)
                game_results_ocr["hit_count_used_for_summary"] = ocr_hit_count
            else:
                game_results_ocr["hit_count_used_for_summary"] = len(df_hits)
        else:
            game_results_ocr["hit_count_used_for_summary"] = len(df_hits)

    summary_data_list = []
    if not df_hits.empty:   # If df_hits is empty, that means no hits were detected, so do not run the next section of code.
        def safe_stat(func, series):
            clean_series = series.dropna()  # Creates a new series from series without any NaN values
            return func(clean_series) if not clean_series.empty else float('nan')   # Checks if any values are in clean_series after NaN values are dropped

        summary_stats = {
            'total_hits_recorded': game_results_ocr.get("hit_count_used_for_summary", 0),
            'ocr_final_score': game_results_ocr.get("final_score"),
            'ocr_accuracy': game_results_ocr.get("accuracy"),
            'ocr_shots_fired': game_results_ocr.get("shot_count"),

            'avg_time_on_target_s': round(safe_stat(statistics.mean, df_hits['time_on_target_s']), 4),
            'median_time_on_target_s': round(safe_stat(statistics.median, df_hits['time_on_target_s']), 4),
            'avg_time_between_hits_s': round(safe_stat(statistics.mean, df_hits['time_between_hits_s']), 4),
            'median_time_between_hits_s': round(safe_stat(statistics.median, df_hits['time_between_hits_s']), 4),

            'avg_flick_time_s': round(safe_stat(statistics.mean, df_hits['flick_time_s']), 4),
            'median_flick_time_s': round(safe_stat(statistics.median, df_hits['flick_time_s']), 4),
            'avg_adj_time_s': round(safe_stat(statistics.mean, df_hits['adjustment_time_s']), 4),
            'median_adj_time_s': round(safe_stat(statistics.median, df_hits['adjustment_time_s']), 4),

            'avg_speed_at_flick_end_px_f': round(safe_stat(statistics.mean, df_hits['speed_at_flick_end_px_f']), 2),
            'median_speed_at_flick_end_px_f': round(safe_stat(statistics.median, df_hits['speed_at_flick_end_px_f']),
                                                    2),
            'avg_flick_dist_px': round(safe_stat(statistics.mean, df_hits['flick_distance_px']), 2),
            'median_flick_dist_px': round(safe_stat(statistics.median, df_hits['flick_distance_px']), 2),
            'avg_norm_flick_time_s_px': round(safe_stat(statistics.mean, df_hits['norm_flick_time_s_per_px']), 6),
            'median_norm_flick_time_s_px': round(safe_stat(statistics.median, df_hits['norm_flick_time_s_per_px']), 6),
            'avg_flick_speed_px_s': round(safe_stat(statistics.mean, df_hits['avg_flick_speed_px_s']), 2),
            'median_flick_speed_px_s': round(safe_stat(statistics.median, df_hits['avg_flick_speed_px_s']), 2),
        }
        summary_data_list.append(summary_stats)

    df_summary = pd.DataFrame(summary_data_list)

    hits_csv_path = output_csv_dir / f"{video_file_path.stem}_hit_metrics.csv"
    summary_csv_path = output_csv_dir / f"{video_file_path.stem}_summary.csv"

    df_hits.to_csv(hits_csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    print(f"--- Analysis for {video_file_path.name} Complete ---")
    print(f"Hit-specific metrics saved to: {hits_csv_path}")
    print(f"Summary metrics saved to: {summary_csv_path}")
    if config["ENABLE_DEBUG_VISUALIZATION"]:
        print(f"Processed video saved to: {output_video_path}")


# --- BATCH PROCESSING FUNCTION ---
def batch_process_folder(
        input_folder_str: str,
        output_base_folder_str: str,
        cli_config_overrides: Dict  # Pass CLI args as a dict
):
    """
    Processes all videos in a folder AND its subfolders
    """
    input_folder = Path(input_folder_str)
    output_base_folder = Path(output_base_folder_str)

    # Combining default config with CLI overrides, CLI overrides will take precedence
    current_run_config = DEFAULT_APP_CONFIG.copy()
    current_run_config.update({k: v for k, v in cli_config_overrides.items() if v is not None})

    # Ensure path objects are correctly formed if passed as strings
    if isinstance(current_run_config.get("YOLO_MODEL_PATH"), str):
        current_run_config["YOLO_MODEL_PATH"] = Path(current_run_config["YOLO_MODEL_PATH"])

    if not current_run_config["YOLO_MODEL_PATH"].exists():
        print(f"Error: YOLO model not found at {current_run_config['YOLO_MODEL_PATH']}")
        print(
            "Please ensure the model exists at this path, or provide a correct path via --yolo_model.")
        return

    processed_videos_main_dir = output_base_folder / "processed_videos"
    csv_reports_main_dir = output_base_folder / "csv_reports"
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files_to_process = []
    for ext in video_extensions:
        video_files_to_process.extend(list(input_folder.glob(f"**/{ext}")))  # Recursively searches for videos in path

    if not video_files_to_process:
        print(f"No video files found in {input_folder} (or its subdirectories) with extensions: {video_extensions}")
        return

    print(f"Found {len(video_files_to_process)} video(s) to process from {input_folder}.")
    all_videos_summary_data = []

    for video_file in video_files_to_process:
        print(f"\n Starting batch processing for: {video_file.resolve()}")
        try:
            # Constructing output paths
            relative_path_from_input = video_file.parent.relative_to(input_folder)
            current_processed_video_dir = processed_videos_main_dir / relative_path_from_input
            output_video_file = current_processed_video_dir / f"{video_file.stem}_processed{video_file.suffix}"

            current_csv_reports_dir = csv_reports_main_dir / relative_path_from_input / video_file.stem

            main_process_video(
                video_file_path=video_file,
                output_video_path=output_video_file,
                output_csv_dir=current_csv_reports_dir,
                config=current_run_config  # Pass the combined config
            )

            summary_csv_path = current_csv_reports_dir / f"{video_file.stem}_summary.csv"
            if summary_csv_path.exists():
                df_summary = pd.read_csv(summary_csv_path)
                if not df_summary.empty:
                    summary_dict = df_summary.iloc[0].to_dict()
                    summary_dict['video_filename'] = video_file.name  # Add filename for overall summary
                    all_videos_summary_data.append(summary_dict)
            print(f">>> Successfully batch processed: {video_file.name} <<<")

        except FileNotFoundError as e_fnf:  # Catch specific errors
            print(f"FILE NOT FOUND ERROR during batch processing of {video_file.name}: {e_fnf}")
            traceback.print_exc()
            with open(output_base_folder / "batch_errors.log", "a") as f_err:
                f_err.write(f"FileNotFoundError processing {video_file.name}: {e_fnf}\n{traceback.format_exc()}\n---\n")
        except Exception as e:
            print(f"GENERAL ERROR during batch processing of {video_file.name}: {e}")
            traceback.print_exc()
            with open(output_base_folder / "batch_errors.log", "a") as f_err:
                f_err.write(f"Error processing {video_file.name}: {e}\n{traceback.format_exc()}\n---\n")
        print("-" * 50)

    if all_videos_summary_data:
        overall_summary_df = pd.DataFrame(all_videos_summary_data)
        overall_summary_path = output_base_folder / "overall_batch_summary.csv"
        overall_summary_df.to_csv(overall_summary_path, index=False)
        print(f"\nOverall batch summary saved to: {overall_summary_path}")
    print("\nBatch processing finished.")


# --- Script entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process aim trainer videos for performance metrics.")
    parser.add_argument("input_folder", type=str, help="Folder containing videos to process.")
    parser.add_argument("output_folder", type=str,
                        help="Base folder where processed videos and CSV reports will be saved.")

    # Allow overriding key config parameters via CLI
    parser.add_argument("--yolo_model", type=str, default=None,
                        # Default is None, will use DEFAULT_APP_CONFIG if not set
                        help=f"Path to YOLO model. Default: '{DEFAULT_APP_CONFIG['YOLO_MODEL_PATH']}' (relative to script or in 'models/' subdir).")
    parser.add_argument("--no_viz", action="store_false", dest="enable_visualization", default=None,
                        help="Disable debug visualization output videos (faster processing). Visualization is ON by default.")
    parser.add_argument("--flick_radius", type=int, default=None,
                        help=f"Radius (px) for flick proximity. Default: {DEFAULT_APP_CONFIG['FLICK_PROXIMITY_RADIUS']}")
    parser.add_argument("--conf_thresh", type=float, default=None,
                        help=f"YOLO confidence threshold. Default: {DEFAULT_APP_CONFIG['YOLO_CONF_THRESH']}")
    parser.add_argument("--iou_thresh", type=float, default=None,
                        help=f"YOLO IOU threshold. Default: {DEFAULT_APP_CONFIG['YOLO_IOU_THRESH']}")

    args = parser.parse_args()

    # Prepare CLI overrides to pass to batch processing
    cli_overrides = {
        "YOLO_MODEL_PATH": Path(args.yolo_model) if args.yolo_model else DEFAULT_APP_CONFIG["YOLO_MODEL_PATH"],
        "ENABLE_DEBUG_VISUALIZATION": args.enable_visualization if args.enable_visualization is not None else
        DEFAULT_APP_CONFIG["ENABLE_DEBUG_VISUALIZATION"],
        "FLICK_PROXIMITY_RADIUS": args.flick_radius if args.flick_radius is not None else DEFAULT_APP_CONFIG[
            "FLICK_PROXIMITY_RADIUS"],
        "YOLO_CONF_THRESH": args.conf_thresh if args.conf_thresh is not None else DEFAULT_APP_CONFIG[
            "YOLO_CONF_THRESH"],
        "YOLO_IOU_THRESH": args.iou_thresh if args.iou_thresh is not None else DEFAULT_APP_CONFIG["YOLO_IOU_THRESH"],
    }
    # Ensures yolo_model path is Path object
    if isinstance(cli_overrides["YOLO_MODEL_PATH"], str):
        cli_overrides["YOLO_MODEL_PATH"] = Path(cli_overrides["YOLO_MODEL_PATH"])

    batch_process_folder(
        input_folder_str=args.input_folder,
        output_base_folder_str=args.output_folder,
        cli_config_overrides=cli_overrides
    )
