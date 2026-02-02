import cv2 as cv
import numpy as np

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
cv.namedWindow("Dashboard") 
cv.resizeWindow("Dashboard", 1400, 700) # Iets breder voor extra sliders

# KLEUR FILTER Sliders
cv.createTrackbar("H_low", "Dashboard", 0, 179, nothing)
cv.createTrackbar("H_high", "Dashboard", 94, 179, nothing)
cv.createTrackbar("S_low", "Dashboard", 29, 255, nothing)
cv.createTrackbar("V_low", "Dashboard", 97, 255, nothing)

# VORM FILTER Sliders
cv.createTrackbar("AR_low", "Dashboard", 70, 200, nothing)    # x100 (Aspect Ratio ondergrens voor 'vierkant')
cv.createTrackbar("AR_high", "Dashboard", 130, 200, nothing)  # x100 (Aspect Ratio bovengrens voor 'vierkant')
cv.createTrackbar("Circ_min", "Dashboard", 40, 100, nothing)  # x100 (Algemene minimum rondheid)

# NIEUWE CLASSIFICATIE Sliders
# Grens tussen Moer en Ring (Ringen zijn ronder dan zeshoekige moeren)
cv.createTrackbar("Split_Circ", "Dashboard", 85, 100, nothing) # x100. Boven = Ring, Onder = Moer

# Grens tussen Spijker en Schroef (Dikte in pixels)
cv.createTrackbar("Split_Thick", "Dashboard", 25, 100, nothing) # Pixels. Dunner = Spijker, Dikker = Schroef

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
    
    # Nieuwe drempelwaardes
    split_circ_thresh = cv.getTrackbarPos("Split_Circ", "Dashboard") / 100.0
    split_thick_thresh = cv.getTrackbarPos("Split_Thick", "Dashboard")

    # B. Processing
    lower = np.array([H_low, S_low, V_low])
    upper = np.array([H_high, 255, 255])
    mask = cv.inRange(hsv_base, lower, upper)
    
    # Morphologie om gaten te dichten en ruis te verwijderen
    kernel = np.ones((5,5), np.uint8)
    mask_cleaned = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_OPEN, kernel)

    # C. Detectie & Classificatie
    contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    live_detection = img.copy()
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > MIN_AREA:
            # 1. Basis Geometrie
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            perimeter = cv.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            # 2. Geavanceerde Geometrie (Rotated Rectangle voor echte dikte)
            rect = cv.minAreaRect(cnt) # Geeft (center), (width, height), angle
            (rot_w, rot_h) = rect[1]
            # De kleinste dimensie is de 'dikte', ongeacht hoe het object gedraaid is
            thickness = min(rot_w, rot_h)

            # 3. Logica Boom
            
            # Is het 'Vierkant/Rond' (dus een Moer of Ring)?
            is_square_shape = (ar_l < aspect_ratio < ar_h)
            
            label = ""
            color = (0,0,0)
            
            if is_square_shape and circularity > c_min:
                # Het is een Moer of een Ring
                if circularity > split_circ_thresh:
                    label = "Ring"
                    color = (255, 0, 0) # Blauw
                else:
                    label = "Moer"
                    color = (0, 255, 0) # Groen
            else:
                # Het is Langwerpig (Schroef of Spijker)
                if thickness < split_thick_thresh:
                    label = "Spijker"
                    color = (255, 0, 255) # Magenta (Dun)
                else:
                    label = "Schroef" 
                    color = (0, 165, 255) # Oranje (Dikker)

            # 4. Tekenen
            cv.rectangle(live_detection, (x, y), (x + w, y + h), color, 2)
            
            # Label + Debug Info
            cv.putText(live_detection, label, (x, y - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            info_text = f"C:{circularity:.2f} Th:{int(thickness)}"
            cv.putText(live_detection, info_text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- STAP E: BEELDEN SAMENVOEGEN ---
    mask_bgr = cv.cvtColor(mask_cleaned, cv.COLOR_GRAY2BGR)
    
    cv.putText(mask_bgr, "Masker (Cleaned)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv.putText(live_detection, "Resultaat", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    combined_view = np.hstack((mask_bgr, live_detection))
    
    # Schalen
    scale_factor = 0.5
    width_rs = int(combined_view.shape[1] * scale_factor)
    height_rs = int(combined_view.shape[0] * scale_factor)
    combined_resized = cv.resize(combined_view, (width_rs, height_rs))

    cv.imshow("Dashboard", combined_resized)

    if cv.waitKey(1) & 0xFF == 27: # ESC
        break

cv.destroyAllWindows()