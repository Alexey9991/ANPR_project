from ultralytics import YOLO
import cv2
import torch
import argparse
import easyocr
import os
import numpy as np

def main(video_path):
    print(torch.__version__)
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    try:
        vehicle_model = YOLO('yolov8n.pt').to(device)
        plate_model = YOLO('license_plate_detector.pt').to(device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print("Models loaded successfully.")
    current_directory = os.getcwd()
    weight_directory = os.path.join(current_directory, 'weight')

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], model_storage_directory=weight_directory, gpu=torch.cuda.is_available())

    def yolo_detection(model, frame):
        # Убираем preprocess_frame_with_padding
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            results = model(frame_tensor)[0]
        return results.boxes.data.tolist()

    def recognize_plate_text(cropped_plate):
        if cropped_plate is None or cropped_plate.size == 0:
            return None
        results = reader.readtext(cropped_plate)
        if results and len(results) > 0:
            return results[0][1]
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    print("Video opened successfully.")
    frame_skip = 10
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        try:
            vehicle_boxes = yolo_detection(vehicle_model, frame)
            for vehicle_box in vehicle_boxes:
                x1, y1, x2, y2, score, class_id = map(int, vehicle_box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Vehicle {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                vehicle_plate = frame[y1:y2, x1:x2]

                plate_boxes = yolo_detection(plate_model, vehicle_plate)

                for plate_box in plate_boxes:
                    px1, py1, px2, py2, p_score, p_class = map(int, plate_box)
                    px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1

                    cropped_plate = vehicle_plate[py1:py2, px1:px2]
                    plate_text = recognize_plate_text(cropped_plate)

                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

                    if plate_text:
                        cv2.putText(frame, plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(f"Detected plate text: {plate_text}")

        except Exception as e:
            print(f"Detection failed: {e}")

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()

    main(args.video_path)