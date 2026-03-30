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
    
    # 1. หาขอบนอกด้วย Canny Auto แบบจูนค่าให้กว้าง
    v = np.median(blurred)
    edged = cv2.Canny(blurred, int(max(0, (1.0 - 0.33) * v)), int(min(255, (1.0 + 0.33) * v)))
    edged = cv2.dilate(edged, np.ones((5,5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "หาการ์ดไม่เจอ ลองขยับมุมกล้องครับ"}

    card_cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(card_cnt, True)
    approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)

    centering_text = "Analysis Failed: วางการ์ดให้เห็น 4 มุมชัดเจน"

    if len(approx) == 4:
        rect = order_points(approx.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        # วาดเส้นขอบนอกแบบนิ่งๆ
        cv2.polylines(output_img, [np.int32(rect)], True, (0, 255, 0), 6)

        # ----------------------------------------------------------------------
        # หัวใจใหม่: GrabCut (ลบพื้นหลัง) เพื่อความแม่นยำสูงสุด
        # ----------------------------------------------------------------------
        # สร้าง ROI สำหรับ GrabCut รอบๆ การ์ดที่เจอ
        x_min, y_min = np.int32(np.min(rect, axis=0))
        x_max, y_max = np.int32(np.max(rect, axis=0))
        h_card, w_card = (y_max - y_min), (x_max - x_min)
        
        # รัน GrabCut (5 ครั้ง)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect_gc = (x_min, y_min, w_card, h_card)
        cv2.grabCut(img, mask, rect_gc, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask_fg = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        img_fg = img * mask_fg[:,:,np.newaxis] # ได้รูปการ์ดที่ตัดพื้นหลังออกทั้งหมด

        # ----------------------------------------------------------------------
        # หาขอบในจากรูปที่ตัดพื้นหลังแล้ว (Precision Mode)
        # ----------------------------------------------------------------------
        fg_gray = cv2.cvtColor(img_fg, cv2.COLOR_BGR2GRAY)
        fg_gray = fg_gray[y_min:y_max, x_min:x_max] # โฟกัสเฉพาะตัวการ์ด
        fg_thresh = cv2.adaptiveThreshold(fg_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        inner_cnts, _ = cv2.findContours(fg_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_inner = None
        min_dist = float('inf')
        
        for cnt in inner_cnts:
            ix, iy, iw, ih = cv2.boundingRect(cnt)
            iarea = iw * ih
            # Artwork ปกติจะกินพื้นที่ 40% - 85% ของการ์ด
            if (h_card * w_card * 0.4) < iarea < (h_card * w_card * 0.85):
                # ตรวจสอบว่าอยู่กึ่งกลางของการ์ด ROI ไหม
                dist = abs((ix + iw/2) - w_card/2) + abs((iy + ih/2) - h_card/2)
                if dist < (w_card * 0.2): # ต้องอยู่ไม่ไกลจากกลางการ์ดมาก
                    best_inner = (ix, iy, iw, ih)
                    break

        if best_inner:
            ix, iy, iw, ih = best_inner
            
            # คำนวณ Ratio
            left, right = ix, w_card - (ix + iw)
            top, bottom = iy, h_card - (iy + ih)
            lr_ratio = round((left / (left + right)) * 100) if (left + right) > 0 else 50
            tb_ratio = round((top / (top + bottom)) * 100) if (top + bottom) > 0 else 50
            centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} T/B: {tb_ratio}/{100-tb_ratio}"

            # วาดขอบในสีทอง (VaultLive) ลงบนรูป Raw ด้วยพิกัดที่แปลงแล้ว
            cv2.rectangle(output_img, (ix + x_min, iy + y_min), 
                          (ix + x_min + iw, iy + y_min + ih), (0, 215, 255), 8)
    else:
        # Fallback กรณีหา 4 มุมไม่เจอ
        x, y, w, h = cv2.boundingRect(card_cnt)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 6)
        centering_text = "หาขอบจาง: กรุณาถ่ายใหม่บนพื้นสีเข้มและนิ่งครับ"

    # แปลง Base64
    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }