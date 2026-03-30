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

    # --- Step 1: ค้นหาขอบการ์ดนอก (Outer Edge) ---
    orig_w, orig_h = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # ใช้ Canny ที่ปรับค่าตามความสว่างภาพอัตโนมัติ
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    # ขยายเส้นขอบเล็กน้อยเพื่อให้เชื่อมกัน
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return {"error": "ไม่พบการ์ดในภาพ"}

    # เลือก Contour ที่ใหญ่ที่สุด (ตัวการ์ด)
    card_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_cnt)
    card_area = w * h

    # --- Step 2: ค้นหาขอบ Artwork ใน (Inner Edge) จากรูป RAW ---
    # ใช้ Adaptive Threshold สู้แสงสะท้อนบนรูป RAW
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    inner_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    possible_artworks = []
    card_center_x, card_center_y = x + (w/2), y + (h/2)
    
    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        # เงื่อนไข: ต้องมีพื้นที่ 30% - 85% ของพื้นที่การ์ด และต้องอยู่ใกล้จุดศูนย์กลางการ์ด
        if (card_area * 0.3) < area < (card_area * 0.9):
            # เช็คว่าอยู่กึ่งกลางของการ์ดไหม
            dist = abs((ix + iw/2) - card_center_x) + abs((iy + ih/2) - card_center_y)
            if dist < (w * 0.3): # ต้องอยู่ไม่ไกลจากกลางการ์ดมาก
                possible_artworks.append((ix, iy, iw, ih, dist))

    # เลือกสี่เหลี่ยมด้านในที่ "อยู่ใกล้จุดกึ่งกลางการ์ดที่สุด"
    if possible_artworks:
        best_inner = min(possible_artworks, key=lambda item: item[4])
        ix, iy, iw, ih, _ = best_inner
        
        # --- Step 3: คำนวณ Centering จากรูป RAW ---
        left_dist = ix - x
        right_dist = (x + w) - (ix + iw)
        top_dist = iy - y
        bottom_dist = (y + h) - (iy + ih)
        
        lr_ratio = round((left_dist / (left_dist + right_dist)) * 100) if (left_dist + right_dist) > 0 else 50
        tb_ratio = round((top_dist / (top_dist + bottom_dist)) * 100) if (top_dist + bottom_dist) > 0 else 50

        # --- Step 4: วาดเส้นตรงๆ ลงบนรูปต้นฉบับ (Raw) ---
        output_img = img.copy()
        
        # 1. ขอบนอก (เขียว)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 6)
        # 2. ขอบใน (ทอง VaultLive)
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 8)
        # 3. เส้นกึ่งกลาง (แดง) เพื่อโชว์ความเบี่ยงเบน
        cv2.line(output_img, (int(x+w/2), y), (int(x+w/2), y+h), (0, 0, 255), 2)
        
        centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} T/B: {tb_ratio}/{100-tb_ratio}"
    else:
        output_img = img
        centering_text = "Analysis Failed: กรุณาถ่ายใหม่บนพื้นหลังที่ตัดกัน"

    # แปลงเป็น Base64 ส่งกลับไป
    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }