import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def order_points(pts):
    """ จัดเรียงจุดมุม: tl, tr, br, bl """
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

    # --- 1. หาขอบนอกสุด (Outer Edge) ---
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # ขยายขอบให้เชื่อมกัน
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "ไม่พบการ์ด"}

    # เลือก Contour ที่ใหญ่ที่สุด
    card_cnt = max(contours, key=cv2.contourArea)
    
    # ดัดภาพให้ตรง (Warp Perspective)
    peri = cv2.arcLength(card_cnt, True)
    approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)

    if len(approx) == 4:
        rect = order_points(approx.reshape(4, 2))
        (tl, tr, br, bl) = rect
        # กำหนดขนาดการ์ดมาตรฐาน (Ratio 2.5 x 3.5)
        maxWidth, maxHeight = 500, 700 
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    else:
        # ถ้าหา 4 มุมไม่เจอ ให้ใช้ Crop สี่เหลี่ยมธรรมดา
        x, y, w, h = cv2.boundingRect(card_cnt)
        warped = orig[y:y+h, x:x+w]
        warped = cv2.resize(warped, (500, 700))

    # --- 2. หาขอบใน (Inner Artwork Edge) ---
    # ใช้รูปที่ "ดัดตรงแล้ว" มาหาขอบใน จะแม่นกว่าเดิมมาก
    w, h = 500, 700
    w_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Adaptive Threshold เพื่อสู้กับแสงสะท้อนบนการ์ด
    w_thresh = cv2.adaptiveThreshold(w_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    inner_contours, _ = cv2.findContours(w_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_inner = None
    min_dist_to_center = float('inf')

    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        # Artwork ปกติจะกินพื้นที่ประมาณ 50-85% ของการ์ด
        if (w * h * 0.4) < area < (w * h * 0.9):
            # เลือกตัวที่อยู่ใกล้จุดกึ่งกลางมากที่สุด
            dist = abs((ix + iw/2) - w/2) + abs((iy + ih/2) - h/2)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_inner = (ix, iy, iw, ih)

    # --- 3. คำนวณและวาดเส้น ---
    output_img = warped.copy()
    # วาดขอบนอก (สีเขียว)
    cv2.rectangle(output_img, (0, 0), (w-1, h-1), (0, 255, 0), 10)

    if best_inner:
        ix, iy, iw, ih = best_inner
        l, r = ix, w - (ix + iw)
        t, b = iy, h - (iy + ih)
        
        lr_ratio = round(l / (l + r) * 100) if (l+r) > 0 else 50
        tb_ratio = round(t / (t + b) * 100) if (t+b) > 0 else 50
        
        # วาดขอบใน (สีทอง VaultLive)
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 6)
        centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} T/B: {tb_ratio}/{100-tb_ratio}"
    else:
        centering_text = "Inner Border Not Found"

    # แปลงกลับเป็น Base64
    _, buffer = cv2.imencode('.jpg', output_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{img_b64}"
    }