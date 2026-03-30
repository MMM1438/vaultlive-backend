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

    # --- 1. เตรียมภาพ (เน้นดึงขอบให้ชัดที่สุด) ---
    h_orig, w_orig = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # เพิ่ม Contrast ให้ขอบชัดขึ้น
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # ใช้ Canny แบบกว้างๆ เพื่อให้จับขอบได้ง่ายขึ้น
    edged = cv2.Canny(blurred, 30, 150)
    
    # ขยายเส้นขอบเล็กน้อยเพื่อให้ Contours เชื่อมต่อกัน
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    # --- 2. ค้นหาการ์ด (Outer Border) ---
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"error": "มองไม่เห็นการ์ด ลองวางบนพื้นหลังสีเข้มๆ ครับ"}

    # กรองเอาอันที่ใหญ่ที่สุดและมีสัดส่วนใกล้เคียงการ์ด (2.5x3.5)
    best_card = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h_orig * w_orig * 0.1): continue # ข้ามถ้าเล็กเกินไป
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        
        # สัดส่วนการ์ดปกติคือ ~0.7 หรือ 1.4 (แนวตั้ง/แนวนอน)
        if 0.5 < aspect_ratio < 1.8:
            if area > max_area:
                max_area = area
                best_card = (x, y, w, h)

    if not best_card:
        # Fallback: ถ้ากรองแล้วไม่เจอ ให้เอาอันที่ใหญ่ที่สุดมาเลย
        card_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(card_cnt)
    else:
        x, y, w, h = best_card

    # Crop เฉพาะตัวการ์ด
    card_img = img[y:y+h, x:x+w].copy()
    cw, ch = w, h

    # --- 3. หาขอบใน (Artwork) ---
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # ใช้ Threshold แบบแบ่งขาวดำชัดเจน
    _, thresh = cv2.threshold(card_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # พยายามหากรอบสี่เหลี่ยมด้านใน
    inner_contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    artwork_rect = None
    best_dist = float('inf')
    
    for icnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(icnt)
        iarea = iw * ih
        # Artwork ต้องมีขนาด 30% - 80% ของตัวการ์ด
        if 0.3 * (cw * ch) < iarea < 0.85 * (cw * ch):
            # เลือกตัวที่อยู่ใกล้จุดศูนย์กลางที่สุด
            dist = abs((ix + iw/2) - cw/2) + abs((iy + ih/2) - ch/2)
            if dist < best_dist:
                best_dist = dist
                artwork_rect = (ix, iy, iw, ih)

    # --- 4. สรุปผลและวาดเส้น ---
    output_img = card_img.copy()
    cv2.rectangle(output_img, (0, 0), (cw-2, ch-2), (0, 255, 0), 10) # ขอบนอกสีเขียว
    
    if artwork_rect:
        ix, iy, iw, ih = artwork_rect
        l, r = ix, cw - (ix + iw)
        t, b = iy, ch - (iy + ih)
        
        lr_perc = round(l / (l + r) * 100) if (l+r) > 0 else 50
        tb_perc = round(t / (t + b) * 100) if (t+b) > 0 else 50
        
        # วาดขอบในสีทอง
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 8)
        centering_text = f"L/R: {lr_perc}/{100-lr_perc} T/B: {tb_perc}/{100-tb_perc}"
    else:
        centering_text = "Analysis Failed: กรุณาถ่ายให้ชัดกว่านี้"

    # แปลงเป็น Base64
    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }