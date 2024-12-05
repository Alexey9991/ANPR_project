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
    """Изменяет размер изображения с сохранением пропорций и добавлением отступов."""
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


def main(video_path, db_host, db_name, db_user, db_password):
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
        processed_frame = preprocess_frame_with_padding(frame)
        frame_tensor = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            results = model(frame_tensor)[0]
        return results.boxes.data.tolist()

    def recognize_plate_text(cropped_plate):
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


    connection = connect_to_db(db_host, db_name, db_user, db_password)
    if not connection:
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        try:
            vehicle_boxes = yolo_detection(vehicle_model, frame)

            for box in vehicle_boxes:
                x1, y1, x2, y2, score, class_id = map(int, box)
                vehicle_plate = frame[y1:y2, x1:x2]

                plate_boxes = yolo_detection(plate_model, vehicle_plate)

                for plate_box in plate_boxes:
                    px1, py1, px2, py2, p_score, p_class = map(int, plate_box)
                    px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
                    cropped_plate = vehicle_plate[py1:py2, px1:px2]
                    if cropped_plate.size == 0:
                        print("number plate is too small")
                        continue
                    plate_text = recognize_plate_text(cropped_plate)
                    if plate_text:
                        print(f"Detected plate text: {plate_text}")
                        save_plate_to_db(plate_text, connection)

        except Exception as e:
            print(f"Detection failed: {e}")

    cap.release()
    if connection:
        connection.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--db_host", default="10.251.3.11", help="Database host")
    parser.add_argument("--db_name", required=True, help="Database name") # Database name is required
    parser.add_argument("--db_user", default="alex", help="Database user")
    parser.add_argument("--db_password", default="123456Lsr", help="Database password")
    args = parser.parse_args()

    main(args.video_path, args.db_host, args.db_name, args.db_user, args.db_password)