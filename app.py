import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="Simple AR Photo (Streamlit + MediaPipe)", layout="centered")
st.title("Simple AR Photo — upload or take a photo")
st.write("This demo uses MediaPipe Face Mesh to place a simple sunglasses overlay on detected faces.")

mp_face = mp.solutions.face_mesh

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def apply_simple_sunglasses(cv_img: np.ndarray) -> np.ndarray:
    h, w = cv_img.shape[:2]
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return cv_img  # no face detected

        for face_landmarks in results.multi_face_landmarks:
            # landmark indices for left/right eye outer corners on MediaPipe Face Mesh
            left_outer = face_landmarks.landmark[33]   # left outer
            right_outer = face_landmarks.landmark[263] # right outer
            left_inner = face_landmarks.landmark[133]  # left inner
            right_inner = face_landmarks.landmark[362] # right inner
            nose_tip = face_landmarks.landmark[1]

            # Convert to pixel coords
            def to_px(lm):
                return np.array([int(lm.x * w), int(lm.y * h)])

            p_left_outer = to_px(left_outer)
            p_right_outer = to_px(right_outer)
            p_left_inner = to_px(left_inner)
            p_right_inner = to_px(right_inner)
            p_nose = to_px(nose_tip)

            # Compute center, width, height, and angle for sunglasses
            eye_center = (p_left_outer + p_right_outer) // 2
            eye_width = int(np.linalg.norm(p_right_outer - p_left_outer) * 1.6)
            eye_height = int(eye_width * 0.4)

            # Angle between eyes
            delta = p_right_outer - p_left_outer
            angle = np.degrees(np.arctan2(delta[1], delta[0]))

            # Create overlay with alpha channel
            overlay = np.zeros((h, w, 4), dtype=np.uint8)

            # Build a rotated rectangle for sunglasses (as a filled polygon)
            rect_center = tuple(eye_center)
            box = cv2.boxPoints(((rect_center[0], rect_center[1]), (eye_width, eye_height), angle))
            box = np.int0(box)

            # Draw filled dark shape (sunglass lenses)
            cv2.fillConvexPoly(overlay, box, (10, 10, 10, 220))  # semi-opaque dark

            # Add a small nose bridge
            nose_bridge_w = max(6, eye_width // 12)
            nose_bridge_h = max(6, eye_height // 3)
            bridge_center = (int((p_left_inner[0] + p_right_inner[0]) / 2), int((p_left_inner[1] + p_right_inner[1]) / 2))
            cv2.ellipse(overlay, bridge_center, (nose_bridge_w, nose_bridge_h), angle, 0, 360, (20, 20, 20, 220), -1)

            # Composite overlay onto original image
            bgr = cv_img.copy()
            # split alpha and use blending
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                bgr[:, :, c] = (overlay[:, :, c] * alpha + bgr[:, :, c] * (1 - alpha)).astype(np.uint8)

            cv_img = bgr

    return cv_img

st.sidebar.markdown("### Input")
uploaded = st.sidebar.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
cam_input = st.sidebar.camera_input("Or take a photo (webcam)")

image_source = None
if cam_input is not None:
    image_source = Image.open(cam_input)
elif uploaded is not None:
    image_source = Image.open(uploaded)

if image_source is None:
    st.info("Please upload a photo or take one with your webcam (camera).")
    st.stop()

# Convert and process
img_cv2 = pil_to_cv2(image_source.convert("RGB"))
st.write("Original photo:")
st.image(image_source, use_column_width=True)

with st.spinner("Detecting face and applying AR overlay..."):
    output_cv2 = apply_simple_sunglasses(img_cv2)
    output_pil = cv2_to_pil(output_cv2)

st.write("AR-applied photo:")
st.image(output_pil, use_column_width=True)

st.markdown("---")
st.write("Notes:")
st.write("- This is a minimal demo. For smoother/animated AR on live video, consider using streamlit-webrtc or a native OpenCV window.")
st.write("- To place PNG overlays (e.g., real sunglasses), load the PNG with alpha and compute transform (scale + rotate) to match the face landmarks.")
