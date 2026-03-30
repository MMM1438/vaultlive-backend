import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def order_points(pts):
    """ฟังก์ชันเรียงพิกัด 4 มุม: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid image"}

    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. หาขอบนอกด้วย Canny แบบ Auto
    v = np.median(blurred)
    edged = cv2.Canny(blurred, int(max(0, (1.0 - 0.33) * v)), int(min(255, (1.0 + 0.33) * v)))
    edged = cv2.dilate(edged, np.ones((5,5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "ไม่พบการ์ด"}

    # หา Contour ที่ใหญ่ที่สุด และแปลงเป็น 4 มุม
    card_cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(card_cnt, True)
    approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)

    centering_text = "Analysis Failed: วางการ์ดให้เห็น 4 มุมชัดเจน"

    # ถ้าเจอ 4 มุมเป๊ะๆ (นี่คือสิ่งที่ Pro App ทำ)
    if len(approx) == 4:
        # --- ขั้นที่ 1: ดัดภาพให้ตรง (Warp) ---
        rect = order_points(approx.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        # วาดเส้นขอบนอกลงบนรูป Raw
        cv2.polylines(output_img, [np.int32(rect)], True, (0, 255, 0), 6)

        # กำหนดขนาด Canvas จำลองเพื่อวัดสัดส่วน (ใช้สัดส่วน 2.5 x 3.5)
        maxWidth, maxHeight = 500, 700
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        
        # สร้าง Matrix M (สำหรับดัดตรง) และ M_inv (สำหรับแปลงกลับ)
        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        
        warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))

        # --- ขั้นที่ 2: หาขอบใน (Artwork) จากรูปที่แบนราบ 100% ---
        # บนรูปที่ตรงแล้ว การหาขอบในจะง่ายและแม่นยำมาก
        w_thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        inner_cnts, _ = cv2.findContours(w_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_inner = None
        min_dist = float('inf')
        
        for cnt in inner_cnts:
            ix, iy, iw, ih = cv2.boundingRect(cnt)
            area = iw * ih
            # เช็คว่าขนาดใกล้เคียง Artwork ไหม (40% - 85% ของ 500x700)
            if (350000 * 0.4) < area < (350000 * 0.85):
                # หาตัวที่อยู่ใกล้จุดกึ่งกลางที่สุด
                dist = abs((ix + iw/2) - maxWidth/2) + abs((iy + ih/2) - maxHeight/2)
                if dist < min_dist:
                    min_dist = dist
                    best_inner = (ix, iy, iw, ih)

        # --- ขั้นที่ 3: คำนวณและแปลงพิกัดกลับไปวาดบนรูป Raw ---
        if best_inner:
            ix, iy, iw, ih = best_inner
            
            # คำนวณ Centering บน Canvas 500x700 (แม่นยำ 100%)
            left, right = ix, maxWidth - (ix + iw)
            top, bottom = iy, maxHeight - (iy + ih)
            lr_ratio = round((left / (left + right)) * 100) if (left + right) > 0 else 50
            tb_ratio = round((top / (top + bottom)) * 100) if (top + bottom) > 0 else 50
            centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} T/B: {tb_ratio}/{100-tb_ratio}"

            # สร้างพิกัด 4 มุมของขอบใน (บน Canvas แบนราบ)
            inner_rect = np.array([
                [ix, iy], [ix + iw, iy], 
                [ix + iw, iy + ih], [ix, iy + ih]
            ], dtype="float32").reshape(-1, 1, 2)

            # ใช้ Inverse Matrix แปลงพิกัดกลับไปสู่มุมมองกล้องเอียงๆ
            transformed_inner = cv2.perspectiveTransform(inner_rect, M_inv)
            
            # วาดเส้นขอบในสีทอง ลงบนรูป Raw ด้วยพิกัดที่แปลงแล้ว
            cv2.polylines(output_img, [np.int32(transformed_inner)], True, (0, 215, 255), 8)
    else:
        # Fallback กรณีหา 4 มุมไม่เจอ
        x, y, w, h = cv2.boundingRect(card_cnt)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 6)
        centering_text = "หา 4 มุมไม่เจอ: กรุณาวางบนพื้นสีเข้มและถ่ายให้เห็นขอบชัดเจน"

    # แปลง Base64
    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }