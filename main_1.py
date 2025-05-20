import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2 as cv
import time
import torch
import numpy as np
from ultralytics import YOLO
from models.models import FastDepthV2
from torchvision import transforms
from PIL import Image

# ─── Setup YOLO and FastDepth ────────────────────────────────────────────────
yolo = YOLO("yolov8n.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastDepthV2()
checkpoint = torch.load(
    r"E:\INTERNSHIP\AI_Projects\envs\obj_detect_env\FastDepth\Weights\FastDepthV2_L1GN_Best.pth",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ─── Video Capture ────────────────────────────────────────────────────────────
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

# ─── State Variables ──────────────────────────────────────────────────────────
previous_time = time.time()
object_detected_time = None
object_locked_time = None
prompted = False
object_locked = False
frame_sent = 0

# This will hold our depth-panel once computed
depth_panel_padded = None

# ─── Main Loop ────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flip & resize for consistent display
    flip_frame = cv.flip(frame, 1)
    resized_Frame = cv.resize(flip_frame, (640, 480))

    # define ROI rectangle
    x, y, w, h = 170, 90, 300, 300
    ROI = resized_Frame[y:y+h, x:x+w]
    cv.rectangle(resized_Frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ── Object Detection inside ROI ───────────────────────────────────────────
    if frame_sent != 1:
        results = yolo.track(ROI, stream=True, verbose=False)
        object_in_rectangle = False
        detected_objects = set()
        depth_maps = {}

        for result in results:
            for box in result.boxes:
                if box.conf[0] < 0.6:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # only count if fully inside ROI
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    object_in_rectangle = True
                    if object_detected_time is None:
                        object_detected_time = time.time()

                cls = int(box.cls[0])
                label = result.names[cls]
                detected_objects.add(label)

                # draw detection box
                cv.rectangle(ROI, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(ROI, f'{label} {box.conf[0]:.2f}',
                           (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # crop and run depth on each detected object
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                crop = ROI[y1c:y2c, x1c:x2c]
                if crop.size:
                    t = transform(crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        d = model(t).squeeze().cpu().numpy()
                    depth_maps[label] = d

        # check lock condition
        if object_in_rectangle and object_detected_time:
            if time.time() - object_detected_time > 1:
                object_locked = True
                cv.rectangle(resized_Frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                if object_locked_time is None:
                    object_locked_time = time.time()
        else:
            # reset if no object
            object_detected_time = None
            object_locked_time = None
            object_locked = False
            prompted = False

        # ── Once locked for >1s, compute and cache depth-panel ─────────────────
        if object_locked and object_locked_time and not prompted \
           and time.time() - object_locked_time > 1:
            prompted = True
            frame_sent = 1
            print("Detected objects:", ", ".join(detected_objects))

            # 1) full-ROI depth
            roi_t = transform(ROI).unsqueeze(0).to(device)
            with torch.no_grad():
                dr = model(roi_t).squeeze().cpu().numpy()
            dr_color = cv.applyColorMap(
                cv.convertScaleAbs(dr, alpha=255.0/(dr.max()+1e-6)),
                cv.COLORMAP_INFERNO
            )

            # 2) pick closest object map
            closest_label = None
            closest_map = None
            min_md = float('inf')
            for lbl, dm in depth_maps.items():
                c = dm[dm.shape[0]//4:3*dm.shape[0]//4,
                       dm.shape[1]//4:3*dm.shape[1]//4].mean()
                if c < min_md:
                    min_md, closest_label, closest_map = c, lbl, dm

            if closest_map is not None:
                cm_color = cv.applyColorMap(
                    cv.convertScaleAbs(closest_map, alpha=255.0/(closest_map.max()+1e-6)),
                    cv.COLORMAP_INFERNO
                )
            else:
                cm_color = np.zeros_like(dr_color)

            # resize everything to 224×224 for panel
            R = cv.resize(ROI, (224, 224))
            D = cv.resize(dr_color, (224, 224))
            C = cm_color

            # add titles
            cv.putText(R, "ROI", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv.putText(D, "ROI Depth", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv.putText(C, f"Closest: {closest_label}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,150,0), 2)

            panel = np.hstack([R, D, C])
            depth_panel_padded = cv.copyMakeBorder(panel, 128, 128, 0, 0, cv.BORDER_CONSTANT, value=(0,0,0)
            )

    # ── Always display combined view 
    # FPS overlay
    now = time.time()
    fps = 1/(now - previous_time)
    previous_time = now
    cv.putText(resized_Frame, f'FPS: {int(fps)}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # choose final_view
    if frame_sent == 1 and depth_panel_padded is not None:
        final_view = np.hstack([resized_Frame, depth_panel_padded])
    else:
        final_view = resized_Frame

    cv.imshow("Object Detection + Depth Views", final_view)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
