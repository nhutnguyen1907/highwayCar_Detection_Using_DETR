from utils import read_video
from utils import save_video
from car import CarDetections
# from car import TruckDetection
import cv2
import pandas as pd
from utils import detect_optimal_highway, get_video_fps, calculate_pixel_to_meter_ratio

def main():
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    highway_coordinates = detect_optimal_highway(video_frames)
    # print("Highway Coordinates:", highway_coordinates)
    highway_length_pixel = 600
    pixel_to_meter_ratio = calculate_pixel_to_meter_ratio(highway_length_pixel)
    
    fps = get_video_fps(input_video_path)
    car_predict = CarDetections(model_path='rtdetr-l.pt', highway_coordinates=highway_coordinates,
                                pixel_to_meter_ratio=pixel_to_meter_ratio, fps=fps)
    # truck_predict = TruckDetection(model_path= 'rtdetr-l.pt')

    car_detections = car_predict.detect_frames(video_frames, read_from_stub=True,
                                                     stub_path="tracker_stubs/car_detections.pkl")
    # truck_detections = truck_predict.detect_frames(video_frames, read_from_stub=False,
    #                                                  stub_path="tracker_stubs/truck_detections.pkl")
    car_detections = car_predict.detect_highway_cars(car_detections)
    car_speeds = car_predict.calculate_car_speed(car_detections)
    print("Car Speeds:", car_speeds)
    output_video_frames = car_predict.draw_bboxes(video_frames, car_detections)
    # output_video_frames = truck_predict.draw_bboxes(output_video_frames, truck_detections)
    
    save_video(output_video_frames,"output_videos/output_video.avi" )

if __name__ == "__main__":
    main()