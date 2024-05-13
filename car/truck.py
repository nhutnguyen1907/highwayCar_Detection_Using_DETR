from ultralytics import RTDETR
import cv2
import pickle
import sys
sys.path.append('../')
# from utils import measurement_dis, get_center_of_bbox

class TruckDetection:
    def __init__ (self, model_path):
        self.model = RTDETR(model_path)
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        truck_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                truck_detections = pickle.load(f)
            return truck_detections

        for frame in frames:
            truck_dict = self.detect_frame(frame)
            truck_detections.append(truck_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(truck_detections, f)
        
        return truck_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        truck_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "truck":
                truck_dict[track_id] = result
        
        return truck_dict

        
    def draw_bboxes(self,video_frames, truck_detections):
        output_video_frames = []
        for frame, truck_dict in zip(video_frames, truck_detections):
            # Draw Bounding Boxes
            for track_id, bbox in truck_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Truck ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
    
    