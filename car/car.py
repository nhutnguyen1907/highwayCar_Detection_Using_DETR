from ultralytics import RTDETR
import cv2
import pickle
import sys
import numpy as np
sys.path.append('../')
from utils import detect_optimal_highway

class CarDetections:
    def __init__ (self, model_path, highway_coordinates, pixel_to_meter_ratio,fps):
        self.model = RTDETR(model_path)
        self.highway_coordinates = highway_coordinates
        self.fps = fps
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.last_frame_car_positions = {}


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        car_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                car_detections = pickle.load(f)
            return car_detections

        for frame in frames:
            car_dict = self.detect_frame(frame)
            car_detections.append(car_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(car_detections, f)
        
        return car_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        car_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "car":
                car_dict[track_id] = result
        
        return car_dict
    
    def detect_highway_cars(self, car_detections):
        highway_cars = []
        for car_dict in car_detections:
            highway_car_dict = {}
            for track_id, bbox in car_dict.items():
                if self.is_on_highway(bbox):
                    highway_car_dict[track_id] = bbox
            highway_cars.append(highway_car_dict)
        return highway_cars

    def is_on_highway(self, bbox):
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        for (x1, y1), (x2, y2) in self.highway_coordinates:
            if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                return True
        return False
        
    def draw_bboxes(self, video_frames, car_detections):
        output_video_frames = []
        for frame, car_dict in zip(video_frames, car_detections):
            for track_id, bbox in car_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Car ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
        
    def calculate_car_speed(self, car_detections):
        car_speeds = {}

        for frame_idx, car_dict in enumerate(car_detections):
            current_frame_car_positions = {}

            for track_id, bbox in car_dict.items():
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                
                # Convert pixel distance to meters
                if track_id in self.last_frame_car_positions:
                    last_x, last_y = self.last_frame_car_positions[track_id]
                    pixel_distance = np.sqrt((x_center - last_x)**2 + (y_center - last_y)**2)
                    distance_in_meters = pixel_distance * self.pixel_to_meter_ratio
                    
                    # Calculate speed in meters per second
                    time_interval = 1.0 / self.fps
                    speed = distance_in_meters / time_interval
                    
                    car_speeds[track_id] = speed
                
                current_frame_car_positions[track_id] = (x_center, y_center)

            self.last_frame_car_positions = current_frame_car_positions
        
        return car_speeds