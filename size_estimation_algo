import cv2 as cv
import numpy as np
import math

# Load image
img = cv.imread(r"E:\INTERNSHIP\AI_Projects\envs\obj_detect_env\FastDepth\experiment\test samples\final\1.png")

# Convert to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Red color range
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Red mask
mask_red = cv.inRange(hsv, lower_red1, upper_red1) | cv.inRange(hsv, lower_red2, upper_red2)

# Find contours
contours, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
red_objects = []
for contour in contours:
    area = cv.contourArea(contour)
    if area > 500:
        red_objects.append(contour)


# Get top 2 red objects
top_2_red = sorted(red_objects, key=cv.contourArea, reverse=True)[:2]

if len(top_2_red) == 2:
    # Reference object (object 0)
    x, y, w, h = cv.boundingRect(top_2_red[0])
    ref_pixel_diagonal = math.sqrt(w**2 + h**2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv.drawContours(img, top_2_red, 0, (0, 255, 0), 2)


    ref_real_diagonal_inch = 2.5  

    # Calculate PPI
    ppi = ref_pixel_diagonal / ref_real_diagonal_inch
    print(f"Reference object: width={w}px, height={h}px, PPI={ppi:.2f}")
    cv.putText(img, f"Reference object , PPI: {ppi:.2f}", (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Target object (object 1)
    tgt_x, tgt_y, tgt_w, tgt_h = cv.boundingRect(top_2_red[1])
    
    cv.rectangle(img, (tgt_x, tgt_y), (tgt_x + tgt_w, tgt_y + tgt_h), (0, 0, 255), 2)
    cv.drawContours(img, top_2_red, 1, (0, 0, 255), 2)
    
    # Calculate target size in height and widht inches
    tgt_pixel_diagonal = math.sqrt(tgt_w**2 + tgt_h**2)
    tgt_real_diagonal_inch = tgt_pixel_diagonal / ppi
    tgt_real_height_inch = (tgt_h / ppi)
    tgt_real_width_inch = (tgt_w / ppi)
    print(f"Target object: width={tgt_w}px, height={tgt_h}px, diagonal={tgt_pixel_diagonal:.2f}px")
    print(f"Target object: width={tgt_real_width_inch:.2f} inches, height={tgt_real_height_inch:.2f} inches, diagonal={tgt_real_diagonal_inch:.2f} inches")
    cv.putText(img, f"Target object width: {tgt_real_width_inch:.2f} inch", (tgt_x, tgt_y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv.putText(img, f"Target object height: {tgt_real_height_inch:.2f} inch", (tgt_x, tgt_y-25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


# Show result
cv.imshow("Detected Red Objects", img)
cv.waitKey(0)
cv.destroyAllWindows()
