import cv2
import numpy as np

import cv2
import numpy as np

def detect_optimal_highway(video_frames):
    highway_coordinates = []

    for frame in video_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                if 280 <= mid_y <= 440:
                    filtered_lines.append(line)

            end_points = [(line[0][2], line[0][3]) for line in filtered_lines]

            if len(end_points) > 0:
                slope, intercept = np.linalg.lstsq(np.vstack([np.array(end_points)[:, 0], np.ones(len(end_points))]).T,
                                                    np.array(end_points)[:, 1], rcond=None)[0]
                x1 = 200
                y1 = int(intercept)
                x2 = 1070
                y2 = int(slope * x2 + intercept)
                highway_coordinates.append(((x1, y1), (x2, y2)))

    return highway_coordinates


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def get_video_fps(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()  # Giải phóng bộ nhớ

    return fps

def calculate_pixel_to_meter_ratio(highway_length_pixel):
    video_highway_length_pixel = highway_length_pixel

    real_highway_length_meter = 1000  

    pixel_to_meter_ratio = real_highway_length_meter / video_highway_length_pixel

    return pixel_to_meter_ratio
