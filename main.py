import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "VaultLive AI Precision Engine is Running"}

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    # --- ขั้นตอนที่ 1: หาขอบการ์ดชั้นนอก (Card Boundary) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # ใช้ Canny ที่ปรับค่าตามความสว่างภาพอัตโนมัติ
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "ไม่พบตัวการ์ดในภาพ"}

    card_cnt = max(contours, key=cv2.contourArea)
    
    # Warp Perspective (ดัดภาพให้ตรง) - ถ้าทำได้จะแม่นขึ้นมาก แต่เบื้องต้นใช้ BoundingRect ก่อน
    x, y, w, h = cv2.boundingRect(card_cnt)
    card_img = img[y:y+h, x:x+w].copy()
    
    # --- ขั้นตอนที่ 2: หาขอบ Artwork ด้านใน (Inner Border) ---
    # ใช้ Adaptive Threshold เพื่อสู้กับแสงสะท้อนบนซองการ์ด
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(card_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Clean Noise เล็กๆ ออกไป
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    inner_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # กรองเฉพาะ Contour ที่มีลักษณะเป็น "สี่เหลี่ยม" และมีขนาด 30% - 80% ของการ์ด
    possible_artworks = []
    card_area = w * h
    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        # เงื่อนไข: ต้องไม่เล็กเกินไป และไม่ใหญ่เท่าตัวการ์ดเอง
        if 0.15 * card_area < area < 0.85 * card_area:
            possible_artworks.append((ix, iy, iw, ih))

    if possible_artworks:
        # เลือกตัวที่อยู่ "กึ่งกลาง" ที่สุด (มักจะเป็น Artwork หลัก)
        center_x, center_y = w / 2, h / 2
        inner_cnt_final = min(possible_artworks, 
                             key=lambda r: (r[0] + r[2]/2 - center_x)**2 + (r[1] + r[3]/2 - center_y)**2)
        ix, iy, iw, ih = inner_cnt_final

        # --- ขั้นตอนที่ 3: คำนวณ Centering แบบละเอียด ---
        # วัดระยะห่างจากขอบการ์ดถึงขอบ Artwork
        left_dist = ix
        right_dist = w - (ix + iw)
        top_dist = iy
        bottom_dist = h - (iy + ih)

        # ป้องกันการหารด้วยศูนย์
        lr_total = left_dist + right_dist if (left_dist + right_dist) > 0 else 1
        tb_total = top_dist + bottom_dist if (top_dist + bottom_dist) > 0 else 1

        lr_ratio = round((left_dist / lr_total) * 100)
        tb_ratio = round((top_dist / tb_total) * 100)

        # วาดเส้นแสดงผล (สีทอง VaultLive)
        cv2.rectangle(card_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 3)
        # วาดเส้นกึ่งกลาง (สีแดง) เพื่อโชว์ความเบี่ยงเบน
        cv2.line(card_img, (int(w/2), 0), (int(w/2), h), (0, 0, 255), 1)
        
        centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} | T/B: {tb_ratio}/{100-tb_ratio}"
    else:
        centering_text = "Analysis Failed: Inner Frame not found"

    # แปลงกลับเป็น Base64
    _, buffer = cv2.imencode('.jpg', card_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{img_base64}",
        "raw_stats": {
            "left": left_dist if possible_artworks else 0,
            "right": right_dist if possible_artworks else 0,
            "top": top_dist if possible_artworks else 0,
            "bottom": bottom_dist if possible_artworks else 0
        }
    }