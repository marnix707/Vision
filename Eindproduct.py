import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# --- INSTELLINGEN ---
MIN_AREA = 500  # Negeer kleine ruis

img = cv.imread(r"media\Image__1.bmp")
if img is None:
    print("Error loading image")
    exit()

# --- 1. Enhancement (Vooraf) ---
gauss = cv.GaussianBlur(img, (5, 5), 0)
kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
sharp = cv.filter2D(gauss, -1, kernel_sharp)
hsv_base = cv.cvtColor(sharp, cv.COLOR_BGR2HSV)

def nothing(x):
    pass

# --- 2. GUI & Trackbars ---
cv.namedWindow("Dashboard") # We gebruiken nu één groot venster
cv.resizeWindow("Dashboard", 1200, 600) # Pas dit aan aan je schermgrootte

cv.createTrackbar("H_low", "Dashboard", 0, 179, nothing)
cv.createTrackbar("H_high", "Dashboard", 94, 179, nothing)
cv.createTrackbar("S_low", "Dashboard", 29, 255, nothing)
cv.createTrackbar("V_low", "Dashboard", 97, 255, nothing)

# CLASSIFICATIE Sliders
cv.createTrackbar("AR_low", "Dashboard", 70, 200, nothing)    # x100
cv.createTrackbar("AR_high", "Dashboard", 130, 200, nothing)  # x100
cv.createTrackbar("Circ_min", "Dashboard", 60, 100, nothing)  # x100

print("Druk op 'ESC' om te stoppen.")

# --- 3. LIVE LOOP ---
while True:
    # A. Lees Trackbars
    H_low = cv.getTrackbarPos("H_low", "Dashboard")
    H_high = cv.getTrackbarPos("H_high", "Dashboard")
    S_low = cv.getTrackbarPos("S_low", "Dashboard")
    V_low = cv.getTrackbarPos("V_low", "Dashboard")
    
    ar_l = cv.getTrackbarPos("AR_low", "Dashboard") / 100.0
    ar_h = cv.getTrackbarPos("AR_high", "Dashboard") / 100.0
    c_min = cv.getTrackbarPos("Circ_min", "Dashboard") / 100.0

    # B. Processing
    lower = np.array([H_low, S_low, V_low])
    upper = np.array([H_high, 255, 255])
    mask = cv.inRange(hsv_base, lower, upper)
    processed_img = cv.bitwise_and(sharp, sharp, mask=mask)

    gray_result = cv.cvtColor(processed_img, cv.COLOR_BGR2GRAY)
    _, thresh_otsu = cv.threshold(gray_result, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    closing = cv.morphologyEx(thresh_otsu, cv.MORPH_CLOSE, kernel)

    # C. Detectie & Classificatie
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    live_detection = img.copy()
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > MIN_AREA:
            x, y, w, h = cv.boundingRect(cnt)
            
            aspect_ratio = float(w) / h
            perimeter = cv.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            is_square = (ar_l < aspect_ratio < ar_h)
            is_round = (circularity > c_min)

            if is_square or is_round:
                label = "Moer/Ring"
                color = (0, 255, 0) # Groen
            else:
                label = "Schroef/Spijker"
                color = (0, 165, 255) # Oranje

            cv.rectangle(live_detection, (x, y), (x + w, y + h), color, 2)
            cv.putText(live_detection, label, (x, y - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Debug info
            info_text = f"AR:{aspect_ratio:.2f} C:{circularity:.2f}"
            cv.putText(live_detection, info_text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- STAP E: BEELDEN SAMENVOEGEN (Side-by-Side) ---
    
    # 1. Maak van de zwart-wit threshold een 3-kanaals beeld (BGR) zodat het past bij de kleurenfoto
    closing_bgr = cv.cvtColor(closing, cv.COLOR_GRAY2BGR)
    
    # 2. Zet tekst boven de beelden voor duidelijkheid
    cv.putText(closing_bgr, "Threshold (Masker)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv.putText(live_detection, "Resultaat (Detectie)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 3. Plak ze naast elkaar (Horizontal Stack)
    combined_view = np.hstack((closing_bgr, live_detection))
    
    # 4. Schaal het beeld eventueel omlaag als het niet op je scherm past (bijv. factor 0.6)
    scale_factor = 0.4
    width = int(combined_view.shape[1] * scale_factor)
    height = int(combined_view.shape[0] * scale_factor)
    combined_resized = cv.resize(combined_view, (width, height))

    # Toon het gecombineerde dashboard
    cv.imshow("Dashboard", combined_resized)

    if cv.waitKey(1) & 0xFF == 27: # ESC
        break

cv.destroyAllWindows()