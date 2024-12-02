from ultralytics import YOLO
import cv2
import torch
import argparse
import psycopg2
import easyocr
import os
import numpy as np


def connect_to_db():
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="plates_db",
            user="user",
            password="password"
        )
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


def save_plate_to_db(plate_text):
    connection = connect_to_db()
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
            connection.close()


def preprocess_frame_with_padding(frame, target_size=640):
    """
    Изменяет размер изображения с сохранением пропорций и добавлением отступов.
    """
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

    # Construct the relative path to the 'weight' directory
    weight_directory = os.path.join(current_directory, 'weight')

    # Initialize EasyOCR
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # Use GPU if available

    def yolo_detection(model, frame):
        # Предварительная обработка кадра
        processed_frame = preprocess_frame_with_padding(frame)

        # Конвертация в тензор
        frame_tensor = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            results = model(frame_tensor)[0]
        return results.boxes.data.tolist()

    def recognize_plate_text(cropped_plate):
        # OCR processing
        results = reader.readtext(cropped_plate)
        if results and len(results) > 0:
            return results[0][1]  # Extract recognized text
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
            # Detect vehicles
            vehicle_boxes = yolo_detection(vehicle_model, frame)

            for box in vehicle_boxes:
                x1, y1, x2, y2, score, class_id = map(int, box)
                vehicle_plate = frame[y1:y2, x1:x2]

                # Detect plates
                plate_boxes = yolo_detection(plate_model, vehicle_plate)

                for plate_box in plate_boxes:
                    px1, py1, px2, py2, p_score, p_class = map(int, plate_box)
                    cropped_plate = vehicle_plate[py1:py2, px1:px2]

                    # Recognize text using EasyOCR
                    plate_text = recognize_plate_text(cropped_plate)
                    if plate_text:
                        print(f"Detected plate text: {plate_text}")
                        save_plate_to_db(plate_text)

        except Exception as e:
            print(f"Detection failed: {e}")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()

    main(args.video_path)
