import cv2 as cv
import numpy as np

# --- INSTELLINGEN ---
MIN_AREA = 2000  # Filter voor kleine ruis op HD beeld

# Start Webcam op HD
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Kan de webcam niet openen.")
    exit()

def nothing(x):
    pass

# --- GUI SETUP ---
cv.namedWindow("Dashboard") 
cv.resizeWindow("Dashboard", 1400, 800)

# 1. KLEUR TRACKBARS
cv.createTrackbar("H_low", "Dashboard", 0, 179, nothing)
cv.createTrackbar("H_high", "Dashboard", 179, 179, nothing)
cv.createTrackbar("S_low", "Dashboard", 50, 255, nothing)
cv.createTrackbar("V_low", "Dashboard", 50, 255, nothing)

# 2. VORM TRACKBARS
cv.createTrackbar("AR_low", "Dashboard", 70, 200, nothing)   # Aspect Ratio Min
cv.createTrackbar("AR_high", "Dashboard", 130, 200, nothing) # Aspect Ratio Max
cv.createTrackbar("Circ_min", "Dashboard", 40, 100, nothing) # Circularity Min

# 3. CLASSIFICATIE TRACKBARS
# Onderscheid Moer vs Ring (Rondheid)
cv.createTrackbar("Split_Circ", "Dashboard", 85, 100, nothing) 
# Onderscheid Spijker vs Schroef (Ruwheid / Karteling)
# Waarde gaat van 1.00 tot 1.50 (x100 in slider)
cv.createTrackbar("Split_Rough", "Dashboard", 108, 150, nothing) 

print("Bediening: 'f' = Freeze | 'ESC' = Stop")

frozen = False
current_frame = None

# --- LIVE LOOP ---
while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame.copy()
    
    img = current_frame.copy()

    # --- A. Enhancement ---
    # Sterke blur voor HD ruis
    gauss = cv.GaussianBlur(img, (9, 9), 0)
    
    # Mild verscherpen om de kartels van de schroefdraad beter te zien
    kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    sharp = cv.filter2D(gauss, -1, kernel_sharp)
    
    hsv = cv.cvtColor(sharp, cv.COLOR_BGR2HSV)

    # --- B. Sliders Uitlezen ---
    H_low = cv.getTrackbarPos("H_low", "Dashboard")
    H_high = cv.getTrackbarPos("H_high", "Dashboard")
    S_low = cv.getTrackbarPos("S_low", "Dashboard")
    V_low = cv.getTrackbarPos("V_low", "Dashboard")
    
    ar_l = cv.getTrackbarPos("AR_low", "Dashboard") / 100.0
    ar_h = cv.getTrackbarPos("AR_high", "Dashboard") / 100.0
    c_min = cv.getTrackbarPos("Circ_min", "Dashboard") / 100.0
    
    split_circ_thresh = cv.getTrackbarPos("Split_Circ", "Dashboard") / 100.0
    # Ruwheid drempel (bijv. 1.08)
    split_rough_thresh = cv.getTrackbarPos("Split_Rough", "Dashboard") / 100.0

    # --- C. Masking ---
    lower = np.array([H_low, S_low, V_low])
    upper = np.array([H_high, 255, 255])
    mask = cv.inRange(hsv, lower, upper)
    
    # Ellipse kernel voor gladdere vormen
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask_cleaned = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_OPEN, kernel)

    # --- D. Detectie ---
    contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    live_detection = img.copy()
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > MIN_AREA:
            # 1. Basis Geometrie
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # Omtrek (Perimeter)
            perimeter = cv.arcLength(cnt, True)
            
            # Rondheid (Circularity)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            # 2. Geavanceerd: Convex Hull & Ruwheid
            hull = cv.convexHull(cnt)
            hull_perimeter = cv.arcLength(hull, True)
            
            # Ruwheid Ratio: Hoeveel langer is de echte omtrek vs het elastiekje?
            # Glad object (Spijker) ≈ 1.0
            # Kartelig object (Schroef) > 1.1
            roughness = perimeter / hull_perimeter if hull_perimeter > 0 else 1.0

            # 3. Logica Boom
            is_square_shape = (ar_l < aspect_ratio < ar_h)
            
            label = "Onbekend"
            color = (100, 100, 100)
            
            if is_square_shape and circularity > c_min:
                # Het is een Moer of Ring
                if circularity > split_circ_thresh:
                    label = "Ring"
                    color = (255, 0, 0) # Blauw
                else:
                    label = "Moer"
                    color = (0, 255, 0) # Groen
            else:
                # Het is Langwerpig (Spijker of Schroef)
                # Hier gebruiken we nu de Ruwheid (Roughness)
                if roughness > split_rough_thresh:
                    label = "Schroef"
                    color = (0, 165, 255) # Oranje (veel kartels)
                else:
                    label = "Spijker" 
                    color = (255, 0, 255) # Magenta (glad)

            # 4. Tekenen
            # Teken de Convex Hull (Het elastiekje) in grijs ter referentie
            cv.drawContours(live_detection, [hull], -1, (180, 180, 180), 1, cv.LINE_AA)
            
            # Teken de Bounding Box
            cv.rectangle(live_detection, (x, y), (x + w, y + h), color, 3)
            
            # Label
            cv.putText(live_detection, label, (x, y - 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Debug Info: Toon de berekende Ruwheid (R)
            info_text = f"C:{circularity:.2f} R:{roughness:.2f}"
            cv.putText(live_detection, info_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # --- E. Weergave ---
    mask_bgr = cv.cvtColor(mask_cleaned, cv.COLOR_GRAY2BGR)
    status_text = "STATUS: BEVROREN" if frozen else "STATUS: LIVE"
    cv.putText(live_detection, status_text, (30, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    combined_view = np.hstack((mask_bgr, live_detection))
    
    # Schalen naar schermbreedte
    target_width = 1400
    scale = target_width / combined_view.shape[1]
    width_rs = int(combined_view.shape[1] * scale)
    height_rs = int(combined_view.shape[0] * scale)
    
    combined_resized = cv.resize(combined_view, (width_rs, height_rs), interpolation=cv.INTER_AREA)

    cv.imshow("Dashboard", combined_resized)

    key = cv.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('f'): 
        frozen = not frozen

cap.release()
cv.destroyAllWindows()