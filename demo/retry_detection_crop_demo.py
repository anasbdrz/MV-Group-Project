import cv2
import os
import glob
import torch
from ultralytics import YOLO

class ALPRGlobalThenVehicleRetry:
    def __init__(self):
        print("--> Loading Vehicle Detector (YOLOv11s)...")
        self.vehicle_model = YOLO("yolo11s.pt")

        print("--> Loading Plate Detector...")
        self.plate_model = YOLO(
            r"C:\MACHINE VISION\ALPR_GPT\runs\detect\my_plate_model_gptv3\weights\best.pt"
        )

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def run(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        plate_crop_dir = os.path.join(output_folder, "plate_crops")
        vehicle_crop_dir = os.path.join(output_folder, "vehicle_crops")
        os.makedirs(plate_crop_dir, exist_ok=True)
        os.makedirs(vehicle_crop_dir, exist_ok=True)

        image_files = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))

        print(f"--> Processing {len(image_files)} images")

        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            used_global_plates = []
            plate_id = 0
            vehicle_id = 0

            with torch.no_grad():
                vehicle_results = self.vehicle_model(
                    frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False
                )
                plate_results_global = self.plate_model(
                    frame, conf=0.25, verbose=False
                )

            global_plates = [list(map(int, p.xyxy[0])) for p in plate_results_global[0].boxes]

            # ---- VEHICLE LOOP ----
            for v_box in vehicle_results[0].boxes:
                vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                vehicle_box = [vx1, vy1, vx2, vy2]

                car_crop = frame[vy1:vy2, vx1:vx2]
                if car_crop.size == 0:
                    continue

                # Save vehicle crop
                car_name = f"img_{idx}_vehicle_{vehicle_id}.jpg"
                cv2.imwrite(os.path.join(vehicle_crop_dir, car_name), car_crop)
                vehicle_id += 1

                matched_plate = None

                # Match global plates
                for g_plate in global_plates:
                    if self.iou(vehicle_box, g_plate) > 0.1:
                        matched_plate = g_plate
                        break

                # Retry inside vehicle
                if matched_plate is None:
                    with torch.no_grad():
                        retry_results = self.plate_model(car_crop, conf=0.15, verbose=False)

                    for p in retry_results[0].boxes:
                        px1, py1, px2, py2 = map(int, p.xyxy[0])
                        matched_plate = [
                            vx1 + px1, vy1 + py1,
                            vx1 + px2, vy1 + py2
                        ]
                        break

                # Draw + save plate
                if matched_plate is not None:
                    x1, y1, x2, y2 = matched_plate
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    used_global_plates.append(matched_plate)

                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    h, w = plate_crop.shape[:2]
                    if w < 120 or h < 30:
                        continue

                    plate_crop = cv2.resize(plate_crop, (320, 80))
                    crop_name = f"img_{idx}_plate_{plate_id}.jpg"
                    cv2.imwrite(os.path.join(plate_crop_dir, crop_name), plate_crop)
                    plate_id += 1

            # ---- FINAL GLOBAL FALLBACK ----
            for g_plate in global_plates:
                if any(self.iou(g_plate, used) > 0.3 for used in used_global_plates):
                    continue

                x1, y1, x2, y2 = g_plate
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                plate_crop = cv2.resize(plate_crop, (320, 80))
                cv2.imwrite(
                    os.path.join(plate_crop_dir, f"img_{idx}_global_fallback.jpg"),
                    plate_crop
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.imwrite(os.path.join(output_folder, f"final_{idx}.jpg"), frame)
            print(f"[{idx+1}/{len(image_files)}] Saved")

        print("\n✅ VEHICLE + PLATE CROPPING PIPELINE COMPLETE")

if __name__ == "__main__":
    app = ALPRGlobalThenVehicleRetry()

    input_dir = r"C:\MACHINE VISION\ALPR_GPT\datasets\test_drhasan"
    output_dir = r"C:\MACHINE VISION\ALPR_GPT\results_global_vehicle_retry"

    app.run(input_dir, output_dir)
