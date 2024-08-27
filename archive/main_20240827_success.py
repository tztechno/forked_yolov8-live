import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(thickness=2)

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    
    # Using supervision's Color class to create the color
    color = Color(r=0, g=0, b=255)  # Red color in BGR format
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=color, thickness=2)

    while True:
        ret, frame = cap.read()

        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Label each detected object
        for i, (xyxy, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
            label = f"{model.names[class_id]} {confidence:0.2f}"
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
