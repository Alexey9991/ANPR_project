from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import cv2
import torch
import argparse
import psycopg2
import easyocr
import os
import numpy as np

def connect_to_db(host, database, user, password):
    try:
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            client_encoding='utf8'
        )
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def save_plate_to_db(plate_text, connection):
    if connection:
        try:
            cursor = connection.cursor()
            insert_query = "INSERT INTO plates (plate_text) VALUES (%s);"
            cursor.execute(insert_query, (plate_text,))
            connection.commit()
            print(f"Number plate '{plate_text}' saved to database.")
        except Exception as e:
            print(f"Error saving to database: {e}")
        finally:
            cursor.close()

def preprocess_frame_with_padding(frame, target_size=640):
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized_frame = cv2.resize(frame, (nw, nh))

    top = (target_size - nh) // 2
    bottom = target_size - nh - top
    left = (target_size - nw) // 2
    right = target_size - nw - left

    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=(114, 114, 114))
    return padded_frame

def process_video(video_path, db_host, db_name, db_user, db_password, device, vehicle_model, plate_model, reader):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    connection = connect_to_db(db_host, db_name, db_user, db_password)
    if not connection:
        return

    frame_skip = 10
    frame_count = 0

    def yolo_detection(frame):
        results = vehicle_model(frame)[0]
        coordinates = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in [2, 3, 5, 7]:
                coordinates.append((x1, y1, x2, y2))
        return coordinates if coordinates else None

    def number_plate_detection(frame):
        frame = cv2.resize(frame, (640, 640))
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
                vehicle_plate = frame[y1:y2, x1:x2]
                plate_text, plate_box = number_plate_detection(vehicle_plate)

                if plate_text and plate_box:
                    save_plate_to_db(plate_text, connection)
                    print(f"Detected number plate: {plate_text}")

    cap.release()
    if connection:
        connection.close()

def main(video_paths, db_host, db_name, db_user, db_password):
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

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_video, video_path, db_host, db_name, db_user, db_password, device, vehicle_model, plate_model, reader)
            for video_path in video_paths
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Vehicle and License Plate Detection")
    parser.add_argument("video_paths", nargs='+', help="Paths to the video files")
    parser.add_argument("--db_host", default="10.251.3.11", help="Database host")
    parser.add_argument("--db_name", required=True, help="Database name")
    parser.add_argument("--db_user", default="alex", help="Database user")
    parser.add_argument("--db_password", default="123456Lsr", help="Database password")
    args = parser.parse_args()

    main(args.video_paths, args.db_host, args.db_name, args.db_user, args.db_password)
