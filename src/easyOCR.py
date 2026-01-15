import cv2
import os
import re
import torch
import numpy as np
import easyocr
from ultralytics import YOLO

class ALPRAdvancedDashboard:
    def __init__(self, vehicle_model_path, plate_model_path, sr_model_path, use_gpu=True):
        print("--> Loading Models...")
        # 1. Detectors
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        
        # 2. EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # 3. Super Resolution (EDSR)
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(sr_model_path)
        self.sr.setModel("edsr", 4)

    @staticmethod
    def iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(areaA + areaB - interArea)

    def postprocess_plate(self, text):
        if not text: return None
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text[:9]

    def enhance_crop(self, crop):
        """Upscale crop 4x using EDSR."""
        try:
            return self.sr.upsample(crop)
        except:
            return cv2.resize(crop, (320, 80))

    def create_dashboard(self, main_img, plate_results):
        h, w, _ = main_img.shape
        panel_w = 400
        panel = np.full((h, panel_w, 3), (25, 25, 25), dtype=np.uint8)
        
        cv2.putText(panel, f"PLATES FOUND: {len(plate_results)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        y_off = 70
        for crop, text in plate_results:
            if y_off + 160 > h: break
            
            # Draw crop with green border
            disp_crop = cv2.resize(crop, (panel_w - 60, 110))
            cv2.rectangle(disp_crop, (0, 0), (disp_crop.shape[1]-1, disp_crop.shape[0]-1), (0, 255, 0), 4)
            
            panel[y_off : y_off + 110, 30 : 30 + (panel_w - 60)] = disp_crop
            cv2.putText(panel, f"OCR: {text}", (35, y_off + 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_off += 180

        return np.hstack((main_img, panel))

    def process_image(self, img_path, out_path):
        frame = cv2.imread(img_path)
        if frame is None: return

        # Stage 1: Global Detections
        with torch.no_grad():
            v_res = self.vehicle_model(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)
            p_res_global = self.plate_model(frame, conf=0.25, verbose=False)

        global_plates = [list(map(int, p.xyxy[0])) for p in p_res_global[0].boxes]
        used_plates = []
        final_plate_data = []

        # Stage 2: Vehicle Loop & Match/Retry
        for v_box in v_res[0].boxes:
            vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
            matched_plate = None

            # Try matching with global plate
            for gp in global_plates:
                if self.iou([vx1, vy1, vx2, vy2], gp) > 0.1:
                    matched_plate = gp
                    break

            # Retry: If no plate found, look inside vehicle crop with lower threshold
            if matched_plate is None:
                car_crop = frame[vy1:vy2, vx1:vx2]
                if car_crop.size > 0:
                    retry_res = self.plate_model(car_crop, conf=0.15, verbose=False)
                    for p in retry_res[0].boxes:
                        px1, py1, px2, py2 = map(int, p.xyxy[0])
                        matched_plate = [vx1+px1, vy1+py1, vx1+px2, vy1+py2]
                        break

            # Stage 3: Enhance, OCR, and Visualize
            if matched_plate:
                x1, y1, x2, y2 = matched_plate
                used_plates.append(matched_plate)
                
                # Crop and Super-Resolve
                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop.size == 0: continue
                enhanced = self.enhance_crop(raw_crop)
                
                # EasyOCR
                ocr_out = self.reader.readtext(enhanced, detail=0)
                txt = self.postprocess_plate(" ".join(ocr_out)) or "UNREADABLE"
                
                final_plate_data.append((enhanced, txt))
                self.draw_label(frame, (x1, y1, x2, y2), txt)

        # Stage 4: Global Fallback (Plates not associated with a specific vehicle)
        for gp in global_plates:
            if not any(self.iou(gp, used) > 0.5 for used in used_plates):
                raw_crop = frame[gp[1]:gp[3], gp[0]:gp[2]]
                if raw_crop.size == 0: continue
                enhanced = self.enhance_crop(raw_crop)
                ocr_out = self.reader.readtext(enhanced, detail=0)
                txt = self.postprocess_plate(" ".join(ocr_out)) or "UNREADABLE"
                
                final_plate_data.append((enhanced, txt))
                self.draw_label(frame, gp, txt, color=(0, 255, 255)) # Yellow for fallback

        # Build Dashboard
        dash = self.create_dashboard(frame, final_plate_data)
        cv2.imwrite(out_path, dash)
        print(f"Finished: {os.path.basename(img_path)}")

    def draw_label(self, img, box, text, color=(0, 255, 0)):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - h - 15), (x1 + w + 10, y1), color, -1)
        cv2.putText(img, text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

if __name__ == "__main__":
    # Paths - Update these to your local paths
    V_MODEL = "yolo11s.pt"
    P_MODEL = r"C:\MACHINE VISION\ALPR_GPT\runs\detect\my_plate_model_gptv3\weights\best.pt" #Adjust the path accordingly
    SR_MODEL = r"C:\MACHINE VISION\ALPR_GPT\src\EDSR_x4.pb" #Adjust the path accordingly
    IN_DIR = r"C:\MACHINE VISION\ALPR_GPT\datasets\test_drhasan" #Adjust the path accordingly
    OUT_DIR = r"C:\MACHINE VISION\ALPR_GPT\results_dashboard_advancedv3" #Adjust the path accordingly

    os.makedirs(OUT_DIR, exist_ok=True)
    app = ALPRAdvancedDashboard(V_MODEL, P_MODEL, SR_MODEL)

    for f in os.listdir(IN_DIR):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            app.process_image(os.path.join(IN_DIR, f), os.path.join(OUT_DIR, f"dash_{f}"))