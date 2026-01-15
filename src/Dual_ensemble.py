import cv2
import os
import re
import torch
import numpy as np
import easyocr
import pytesseract
from ultralytics import YOLO

class ALPREnsembleDashboard:
    def __init__(self, vehicle_model_path, plate_model_path, sr_model_path, use_gpu=True):
        print("--> Loading Models & Engines...")
        # 1. Detectors
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        
        # 2. OCR Engines
        self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        # Tesseract usually found in PATH; uncomment if needed:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
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

    def clean_text(self, text):
        if not text: return ""
        text = text.upper()
        return re.sub(r"[^A-Z0-9]", "", text)[:9]

    def get_ensemble_ocr(self, crop):
        """Runs both OCRs and decides the accepted result."""
        # --- EasyOCR ---
        # Returns list of (bbox, text, confidence)
        easy_results = self.easy_reader.readtext(crop)
        easy_txt = ""
        easy_conf = 0.0
        if easy_results:
            easy_txt = self.clean_text(" ".join([res[1] for res in easy_results]))
            easy_conf = np.mean([res[2] for res in easy_results])

        # --- Tesseract ---
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        tess_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        tess_txt = self.clean_text(pytesseract.image_to_string(gray, config=tess_config).strip())

        # --- Decision Logic ---
        # Priority 1: If both agree
        if easy_txt == tess_txt and easy_txt != "":
            accepted = easy_txt
        # Priority 2: If one is empty, take the other
        elif easy_txt != "" and tess_txt == "":
            accepted = easy_txt
        elif tess_txt != "" and easy_txt == "":
            accepted = tess_txt
        # Priority 3: If different, trust EasyOCR if confidence > 0.5
        else:
            accepted = easy_txt if easy_conf > 0.5 else tess_txt
        
        return {
            "easy": easy_txt,
            "easy_conf": f"{easy_conf:.2f}",
            "tess": tess_txt,
            "accepted": accepted if accepted else "NOT_READ"
        }

    def enhance_crop(self, crop):
        try:
            return self.sr.upsample(crop)
        except:
            return cv2.resize(crop, (320, 80))

    def create_dashboard(self, main_img, plate_results):
        h, w, _ = main_img.shape
        panel_w = 450
        panel = np.full((h, panel_w, 3), (20, 20, 20), dtype=np.uint8)
        
        cv2.putText(panel, f"ALPR ENSEMBLE - PLATES: {len(plate_results)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_off = 70
        for crop, ocr_data in plate_results:
            if y_off + 220 > h: break
            
            # Display Crop
            disp_crop = cv2.resize(crop, (panel_w - 60, 100))
            cv2.rectangle(disp_crop, (0,0), (disp_crop.shape[1]-1, disp_crop.shape[0]-1), (0, 255, 0), 3)
            panel[y_off : y_off + 100, 30 : 30 + (panel_w - 60)] = disp_crop
            
            # OCR Data display
            y_text = y_off + 125
            cv2.putText(panel, f"EasyOCR: {ocr_data['easy']} (Conf: {ocr_data['easy_conf']})", (35, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, f"Tesseract: {ocr_data['tess']}", (35, y_text + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Accepted Result (Highlighted)
            cv2.putText(panel, f"ACCEPTED: {ocr_data['accepted']}", (35, y_text + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_off += 210

        return np.hstack((main_img, panel))

    def process_image(self, img_path, out_path):
        frame = cv2.imread(img_path)
        if frame is None: return

        with torch.no_grad():
            v_res = self.vehicle_model(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)
            p_res_global = self.plate_model(frame, conf=0.25, verbose=False)

        global_plates = [list(map(int, p.xyxy[0])) for p in p_res_global[0].boxes]
        used_plates, final_plate_data = [], []

        for v_box in v_res[0].boxes:
            vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
            matched_plate = None

            for gp in global_plates:
                if self.iou([vx1, vy1, vx2, vy2], gp) > 0.1:
                    matched_plate = gp
                    break

            if matched_plate is None:
                car_crop = frame[vy1:vy2, vx1:vx2]
                if car_crop.size > 0:
                    retry_res = self.plate_model(car_crop, conf=0.15, verbose=False)
                    for p in retry_res[0].boxes:
                        px1, py1, px2, py2 = map(int, p.xyxy[0])
                        matched_plate = [vx1+px1, vy1+py1, vx1+px2, vy1+py2]
                        break

            if matched_plate:
                x1, y1, x2, y2 = matched_plate
                used_plates.append(matched_plate)
                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop.size == 0: continue
                enhanced = self.enhance_crop(raw_crop)
                
                # --- ENSEMBLE OCR ---
                ocr_data = self.get_ensemble_ocr(enhanced)
                final_plate_data.append((enhanced, ocr_data))
                self.draw_label(frame, (x1, y1, x2, y2), ocr_data['accepted'])

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
    V_MODEL = "yolo11s.pt"
    P_MODEL = r"C:\MACHINE VISION\ALPR_GPT\runs\detect\my_plate_model_gptv3\weights\best.pt" #Adjust the path accordingly
    SR_MODEL = r"C:\MACHINE VISION\ALPR_GPT\src\EDSR_x4.pb" #Adjust the path accordingly
    IN_DIR = r"C:\MACHINE VISION\ALPR_GPT\datasets\test_drhasan" #Adjust the path accordingly
    OUT_DIR = r"C:\MACHINE VISION\ALPR_GPT\results_ensemble" #Adjust the path accordingly
 
    os.makedirs(OUT_DIR, exist_ok=True)
    app = ALPREnsembleDashboard(V_MODEL, P_MODEL, SR_MODEL)

    for f in os.listdir(IN_DIR):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            app.process_image(os.path.join(IN_DIR, f), os.path.join(OUT_DIR, f"dash_{f}"))