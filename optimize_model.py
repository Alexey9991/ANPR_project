from ultralytics import YOLO
import cv2
import time
import easyocr
from concurrent.futures import ThreadPoolExecutor
import os
import csv
import datetime
import torch
import argparse



def main(video_path):
    print(torch.__version__)
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading vehicle_model")

    current_directory = os.getcwd()

    # Construct the relative path to the 'weight' directory
    weight_directory = os.path.join(current_directory, 'weight')
    try:
        vehicle_model = YOLO('yolov8n.pt', device = 'cuda:0')

        vehicle_model.to(device)
        vehicle_model.export(half=True)
    except Exception as e:
        print(f"Error loading vehicle model: {e}")
        return

    print("Loading plate_model")
    try:
        plate_model = YOLO('license_plate_detector.pt', device = 'cuda:0')
        plate_model.to(device)
    except Exception as e:
        print(f"Error loading plate model: {e}")
        return

    print("Loading reader")
    try:
        reader = easyocr.Reader(['en'], model_storage_directory=weight_directory, gpu=True,  download_enabled=False)
    except Exception as e:
        print(f"Error loading OCR reader: {e}")
        return


    vehicles = [2, 3, 5, 7]
    csv_file_path = "plates.csv"  # Simplified path, creates in current directory


    def save_plate_to_csv(plate_text):
        file_exists = os.path.isfile(csv_file_path)
        try:
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['plate_text', 'timestamp'])
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([plate_text, current_time])
            print(f"Number plate '{plate_text}' saved to CSV file.")
        except Exception as e:
            print(f"Error saving to CSV file: {e}")


    def yolo_detection(frame):
        results = vehicle_model(frame)[0]
        coordinates = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in vehicles:
                coordinates.append((x1, y1, x2, y2))
        return coordinates if coordinates else None

    def number_plate_detection(frame):
        frame = cv2.resize(frame, (640, 640))
        results = plate_model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = map(int, result)  # Correct type conversion
            cropped_image = frame[y1:y2, x1:x2]  # Correct cropping
            try:
                ocr_result = reader.readtext(cropped_image)
                if ocr_result:  # Check if ocr_result is not empty
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

    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            vehicle_boxes = yolo_detection(frame)

            if vehicle_boxes:
                futures = []
                for vehicle_box in vehicle_boxes:
                    x1, y1, x2, y2 = map(int, vehicle_box)

                    vehicle_plate = frame[y1:y2, x1:x2]
                    futures.append(executor.submit(process_vehicle_plate, vehicle_plate))


                for future in futures:
                    plate_text, plate_box = future.result()
                    if plate_text and plate_box:
                        px1, py1, px2, py2 = map(int, plate_box)
                        px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1  # Correct offset
                        print(f"Detected number plate: {plate_text}")
                        save_plate_to_csv(plate_text)


    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle and License Plate Detection")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()

    main(args.video_path)