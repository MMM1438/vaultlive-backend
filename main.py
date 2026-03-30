import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def order_points(pts):
    """ จัดเรียงจุด 4 จุด: top-left, top-right, bottom-right, bottom-left """
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

    # --- Step 1: ค้นหาการ์ดและดัดให้ตรง (Warp) ---
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "หาการ์ดไม่เจอ ลองถ่ายบนพื้นหลังที่ตัดกัน"}
    
    card_cnt = max(contours, key=cv2.contourArea)
    
    # พยายามหาจุดมุม 4 จุด
    peri = cv2.arcLength(card_cnt, True)
    approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)
    
    if len(approx) == 4:
        # ถ้าเจอ 4 มุม ให้ดัดภาพ (Warp) เพื่อความแม่นยำสูงสุด
        pts = approx.reshape(4, 2)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        card_img = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    else:
        # ถ้าหา 4 มุมไม่ชัด ให้ใช้ Crop สี่เหลี่ยมธรรมดา
        x, y, w, h = cv2.boundingRect(card_cnt)
        card_img = orig[y:y+h, x:x+w]

    # --- Step 2: หาขอบใน (Artwork) จากรูปที่ดัดแล้ว ---
    h, w = card_img.shape[:2]
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Threshold แบบปรับตัว (Adaptive) สู้แสงสะท้อน
    thresh = cv2.adaptiveThreshold(card_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    inner_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_inner = None
    card_area = w * h
    # คัดเลือกสี่เหลี่ยมที่ "กึ่งกลาง" และ "ขนาดพอดี"
    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        if 0.25 * card_area < area < 0.80 * card_area:
            # เช็คว่าอยู่กึ่งกลางไหม
            dist_to_center = abs((ix + iw/2) - w/2) + abs((iy + ih/2) - h/2)
            if best_inner is None or dist_to_center < best_inner['dist']:
                best_inner = {'rect': (ix, iy, iw, ih), 'dist': dist_to_center}

    if best_inner:
        ix, iy, iw, ih = best_inner['rect']
        # คำนวณขอบ
        l, r, t, b = ix, w - (ix + iw), iy, h - (iy + ih)
        
        # วาดเส้น (สีทอง VaultLive)
        cv2.rectangle(card_img, (ix, iy), (ix+iw, iy+ih), (0, 215, 255), 4)
        cv2.rectangle(card_img, (0, 0), (w, h), (0, 255, 0), 6) # ขอบนอกสีเขียว
        
        res_text = f"L/R: {round(l/(l+r)*100)}/{100-round(l/(l+r)*100)} T/B: {round(t/(t+b)*100)}/{100-round(t/(t+b)*100)}"
    else:
        res_text = "ไม่พบกรอบด้านใน (Artwork)"

    _, buffer = cv2.imencode('.jpg', card_img)
    return {
        "centering": res_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }