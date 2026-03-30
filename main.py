import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid image"}

    # --- Step 1: หาขอบนอกให้เป๊ะ (Outer Border) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ใช้ Median Blur แทน Gaussian เพื่อลบเม็ดสีรบกวน (Noise) แต่ยังรักษาขอบคมๆ ไว้
    blurred = cv2.medianBlur(gray, 7)
    
    # ใช้ Adaptive Threshold เพื่อหาโครงสร้างหลัก
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "หาการ์ดไม่เจอ"}

    card_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_cnt)
    
    # ตัดรูปเฉพาะการ์ด (Crop)
    card_img = img[y:y+h, x:x+w].copy()
    cw, ch = w, h

    # --- Step 2: หาขอบใน (Logic ใหม่: Multi-Scale Edge Detection) ---
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    
    # ทริค: ใช้ Canny สองรอบเพื่อหาเส้นที่ซ่อนอยู่
    edged = cv2.Canny(card_gray, 50, 200)
    # ขยายเส้นขอบให้เชื่อมกัน
    edged = cv2.dilate(edged, None, iterations=1)
    
    inner_contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    # กรองหา "กรอบรูปภาพ" (Artwork Frame)
    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        # เงื่อนไข: ต้องมีพื้นที่ 40% - 80% ของการ์ด และต้องอยู่ใกล้จุดศูนย์กลาง
        if 0.4 * (cw * ch) < area < 0.85 * (cw * ch):
            dist_to_center = abs((ix + iw/2) - cw/2) + abs((iy + ih/2) - ch/2)
            if best_rect is None or dist_to_center < best_rect['dist']:
                best_rect = {'coords': (ix, iy, iw, ih), 'dist': dist_to_center}

    # --- Step 3: คำนวณ Centering ---
    output_img = card_img.copy()
    if best_rect:
        ix, iy, iw, ih = best_rect['coords']
        
        # วัดระยะห่าง 4 ทิศ (หน่วยเป็น Pixel)
        left = ix
        right = cw - (ix + iw)
        top = iy
        bottom = ch - (iy + ih)

        # คำนวณสูตร Centering: (ระยะฝั่งหนึ่ง / ระยะรวมสองฝั่ง) * 100
        lr_ratio = round((left / (left + right)) * 100) if (left + right) > 0 else 50
        tb_ratio = round((top / (top + bottom)) * 100) if (top + bottom) > 0 else 50

        # วาดเส้น Guide
        # 1. ขอบนอก (เขียว)
        cv2.rectangle(output_img, (0, 0), (cw, ch), (0, 255, 0), 10)
        # 2. ขอบใน (ทอง)
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 8)
        # 3. เส้นกึ่งกลาง (แดง)
        cv2.line(output_img, (int(cw/2), 0), (int(cw/2), ch), (0, 0, 255), 2)
        
        result_text = f"L/R: {lr_ratio}/{100-lr_ratio} | T/B: {tb_ratio}/{100-tb_ratio}"
    else:
        result_text = "Analysis Failed: กรุณาขยับมุมกล้อง"

    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": result_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }