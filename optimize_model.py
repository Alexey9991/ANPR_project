from ultralytics import YOLO
import cv2
import torch
import argparse
import psycopg2
from paddleocr import PaddleOCR
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

    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_gpu=torch.cuda.is_available(), lang='en')

    def yolo_detection(model, frame):
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            results = model(frame_tensor)[0]
        return results.boxes.data.tolist()

    def recognize_plate_text(cropped_plate):
        # Convert cropped image to the format required by PaddleOCR
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
        results = ocr.ocr(cropped_plate, cls=False)
        if results and len(results[0]) > 0:
            return results[0][0][1][0]  # Extract text from results
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

                    # Recognize text using PaddleOCR
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
