import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def order_points(pts):
    """ จัดเรียงจุด 4 มุม: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย """
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

    # --- Step 1: ค้นหาขอบการ์ดและดัดภาพให้ตรง (Warp Perspective) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "ไม่พบตัวการ์ด"}

    card_cnt = max(contours, key=cv2.contourArea)
    
    # พยายามบีบ Contour ให้เหลือ 4 มุม (สี่เหลี่ยม)
    peri = cv2.arcLength(card_cnt, True)
    approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)

    if len(approx) == 4:
        # ดัดภาพให้ตรง (Warp)
        rect = order_points(approx.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        # คำนวณขนาดของการ์ดใหม่หลังจากดัดตรง
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    else:
        # ถ้าหา 4 มุมไม่เจอ ให้ใช้ Crop สี่เหลี่ยมธรรมดา (Fallback)
        x, y, w, h = cv2.boundingRect(card_cnt)
        warped = img[y:y+h, x:x+w]

    # --- Step 2: หาขอบในจากรูปที่ดัดตรงแล้ว (แม่นยำกว่าเดิม 100%) ---
    h, w = warped.shape[:2]
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Adaptive Threshold สู้แสงสะท้อน
    thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    inner_contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    card_area = w * h
    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        # Artwork ต้องมีขนาด 35% - 85% ของการ์ด
        if 0.35 * card_area < area < 0.85 * card_area:
            # เลือกตัวที่อยู่ใกล้จุดศูนย์กลางที่สุด
            dist = abs((ix + iw/2) - w/2) + abs((iy + ih/2) - h/2)
            if best_rect is None or dist < best_rect['dist']:
                best_rect = {'coords': (ix, iy, iw, ih), 'dist': dist}

    # --- Step 3: คำนวณ Centering ---
    output_img = warped.copy()
    if best_rect:
        ix, iy, iw, ih = best_rect['coords']
        # คำนวณระยะห่าง
        l, r = ix, w - (ix + iw)
        t, b = iy, h - (iy + ih)
        
        lr_val = round(l/(l+r)*100) if (l+r)>0 else 50
        tb_val = round(t/(t+b)*100) if (t+b)>0 else 50
        
        # วาดเส้น Guide
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 6) # ขอบในสีทอง
        cv2.rectangle(output_img, (0, 0), (w, h), (0, 255, 0), 10) # ขอบนอกสีเขียว
        
        result_text = f"L/R: {lr_val}/{100-lr_val} | T/B: {tb_val}/{100-tb_val}"
    else:
        result_text = "Analysis Failed: กรุณาวางการ์ดให้ตรงขึ้น"

    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": result_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }