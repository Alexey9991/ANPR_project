from ultralytics import YOLO
import cv2
import torch
import argparse
import easyocr
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

    vehicles = [2, 3, 5, 7] # Классы транспортных средств (car, motorcycle, bus, truck) - подставьте свои значения


    def yolo_detection(frame):
        results = vehicle_model(frame)[0]
        coordinates = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in vehicles:
                coordinates.append((x1, y1, x2, y2))
        return coordinates if coordinates else None

    def number_plate_detection(frame):
        frame = cv2.resize(frame, (640, 640)) #  Resize here
        results = plate_model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = map(int, result)
            cropped_image = frame[y1:y2, x1:x2]
            try:
                ocr_result = reader.readtext(cropped_image)
                if ocr_result:
                    return ocr_result[0][1], (x1, y1, x2, y2)
            except Exception as e:
                print(f"OCR failed: {e}")
        return None, None

    def process_vehicle_plate(vehicle_plate):
        return number_plate_detection(vehicle_plate)


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

        vehicle_boxes = yolo_detection(frame)

        if vehicle_boxes:
            for vehicle_box in vehicle_boxes:
                x1, y1, x2, y2 = map(int, vehicle_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Рисуем рамку вокруг ТС

                vehicle_plate = frame[y1:y2, x1:x2]
                plate_text, plate_box = number_plate_detection(vehicle_plate) # Вызываем функцию напрямую

                if plate_text and plate_box:
                    px1, py1, px2, py2 = map(int, plate_box)
                    px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1  # Correct offset
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)  # Рисуем рамку номера
                    cv2.putText(frame, plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"Detected number plate: {plate_text}")

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