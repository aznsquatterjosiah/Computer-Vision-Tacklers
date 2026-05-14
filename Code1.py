# ============================================
# Names: Josiah Villaflor, Shriram Narendran, Abraham Amanyar
# Student Number: 23389559, 23972901, 23460391
# CITS4402 Project
# ============================================

import os
import time
import shutil
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import mediapipe as mp

# URL for the mediapipe face landmarker model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"

# URLs for OpenCV DNN face detector (res10 SSD)
DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_PROTOTXT_FILE = "deploy.prototxt"
DNN_CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
DNN_CAFFEMODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"


class FaceProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image GUI")
        self.root.configure(bg="#d9d9d9")

        self.box_w = 360
        self.box_h = 260
        self.max_faces_per_image = 4
        self.face_box_color = (255, 220, 120)
        self.face_box_thickness = 2

        self.original_tk = None
        self.output_tk = None

        self.current_input_bgr = None
        self.current_output_bgr = None

        # face detector (OpenCV DNN SSD - more accurate than Haar cascade)
        self.dnn_prototxt, self.dnn_model = self.get_dnn_model_paths()
        self.face_detector = cv2.dnn.readNetFromCaffe(
            str(self.dnn_prototxt), str(self.dnn_model)
        )
        self.dnn_confidence_threshold = 0.45

        # download the face landmarker model if it doesn't exist yet
        self.model_path = self.get_model_path()

        # set up the mediapipe face landmarker (new Tasks API)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # landmark indices in mediapipe's 478-point model
        # 468 = right iris centre, 473 = left iris centre, 1 = nose tip
        self.RIGHT_EYE_IDX = 468
        self.LEFT_EYE_IDX = 473
        self.NOSE_TIP_IDX = 1

        # where we want the landmarks to land in the 125x125 aligned crop
        self.aligned_size = 125
        self.target_landmarks = np.array([
            [40, 40],   # right eye
            [85, 40],   # left eye
            [63, 70]    # nose tip
        ], dtype=np.float32)

        self.build_gui()

    def build_gui(self):
        self.main_frame = tk.Frame(self.root, bg="#d9d9d9", padx=18, pady=18)
        self.main_frame.pack()

        self.top_frame = tk.Frame(self.main_frame, bg="#d9d9d9")
        self.top_frame.grid(row=0, column=0, columnspan=2, pady=(0, 12))

        self.input_canvas = tk.Canvas(
            self.top_frame,
            width=self.box_w,
            height=self.box_h,
            bg="white",
            highlightbackground="black",
            highlightthickness=1
        )
        self.input_canvas.grid(row=0, column=0, padx=(0, 16))
        self.input_canvas.create_text(
            self.box_w // 2, self.box_h // 2, text="Input Image", font=("Arial", 12)
        )

        self.output_canvas = tk.Canvas(
            self.top_frame,
            width=self.box_w,
            height=self.box_h,
            bg="white",
            highlightbackground="black",
            highlightthickness=1
        )
        self.output_canvas.grid(row=0, column=1)
        self.output_canvas.create_text(
            self.box_w // 2, self.box_h // 2, text="Processed Image", font=("Arial", 12)
        )

        self.message_frame = tk.Frame(self.main_frame, bg="#d9d9d9")
        self.message_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.status_var_1 = tk.StringVar(value="Ready.")
        self.status_var_2 = tk.StringVar(value="Load a single image or a folder to begin.")
        self.status_var_3 = tk.StringVar(value="")

        self.status_label_1 = tk.Label(
            self.message_frame, textvariable=self.status_var_1, bg="#d9d9d9", anchor="w", font=("Arial", 11)
        )
        self.status_label_1.pack(anchor="w", pady=(0, 6))

        self.status_label_2 = tk.Label(
            self.message_frame, textvariable=self.status_var_2, bg="#d9d9d9", anchor="w", font=("Arial", 11)
        )
        self.status_label_2.pack(anchor="w", pady=(0, 6))

        self.status_label_3 = tk.Label(
            self.message_frame, textvariable=self.status_var_3, bg="#d9d9d9", anchor="w", font=("Arial", 11)
        )
        self.status_label_3.pack(anchor="w")

        self.button_frame = tk.Frame(self.main_frame, bg="#d9d9d9")
        self.button_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.single_button = tk.Button(
            self.button_frame,
            text="Single Image",
            width=15,
            command=self.load_single_image
        )
        self.single_button.grid(row=0, column=0, padx=(8, 200), pady=8, sticky="w")

        self.bulk_button = tk.Button(
            self.button_frame,
            text="Bulk Processing",
            width=15,
            command=self.bulk_process_folder
        )
        self.bulk_button.grid(row=0, column=1, padx=(120, 8), pady=8, sticky="e")

    def get_model_path(self):
        """Check for the face landmarker model file next to the script,
        and download it from Google if it's not there."""
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / MODEL_FILENAME

        if not model_path.exists():
            print(f"Downloading {MODEL_FILENAME} ...")
            try:
                urllib.request.urlretrieve(MODEL_URL, str(model_path))
                print("Download complete.")
            except Exception as e:
                messagebox.showerror(
                    "Model Download Failed",
                    f"Could not download {MODEL_FILENAME}.\n"
                    f"Please download it manually from:\n{MODEL_URL}\n"
                    f"and place it next to your script.\n\nError: {e}"
                )
                raise

        return model_path

    def get_dnn_model_paths(self):
        """Download the OpenCV DNN face detector files if needed."""
        script_dir = Path(__file__).resolve().parent
        prototxt_path = script_dir / DNN_PROTOTXT_FILE
        caffemodel_path = script_dir / DNN_CAFFEMODEL_FILE

        for url, path, name in [
            (DNN_PROTOTXT_URL, prototxt_path, DNN_PROTOTXT_FILE),
            (DNN_CAFFEMODEL_URL, caffemodel_path, DNN_CAFFEMODEL_FILE)
        ]:
            if not path.exists():
                print(f"Downloading {name} ...")
                try:
                    urllib.request.urlretrieve(url, str(path))
                    print(f"Downloaded {name}.")
                except Exception as e:
                    messagebox.showerror(
                        "Model Download Failed",
                        f"Could not download {name}.\n"
                        f"Please download it manually from:\n{url}\n"
                        f"and place it next to your script.\n\nError: {e}"
                    )
                    raise

        return prototxt_path, caffemodel_path

    def prepare_tk_image(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        scale = min(self.box_w / w, self.box_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb_image, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(pil_img)

    def show_on_canvas(self, canvas, image_bgr, canvas_name):
        canvas.delete("all")
        tk_img = self.prepare_tk_image(image_bgr)
        canvas.create_image(self.box_w // 2, self.box_h // 2, image=tk_img)

        if canvas_name == "input":
            self.original_tk = tk_img
        else:
            self.output_tk = tk_img

    def reset_output_canvas(self):
        self.output_canvas.delete("all")
        self.output_canvas.create_text(
            self.box_w // 2, self.box_h // 2, text="Processed Image", font=("Arial", 12)
        )

    def load_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a single image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not open the selected image.")
            return

        self.current_input_bgr = image.copy()
        self.show_on_canvas(self.input_canvas, self.current_input_bgr, "input")

        start = time.time()
        output_bgr, faces_info = self.process_single_image(image)
        elapsed = time.time() - start

        self.current_output_bgr = output_bgr
        self.show_on_canvas(self.output_canvas, self.current_output_bgr, "output")

        self.status_var_1.set(f"Single image processed in {elapsed:.2f} seconds.")
        self.status_var_2.set(f"Single image found {len(faces_info)} face(s).")
        self.status_var_3.set(f"File: {os.path.basename(file_path)}")

    def bulk_process_folder(self):
        folder_path = filedialog.askdirectory(title="Select folder of input images")
        if not folder_path:
            return

        image_paths = self.list_image_files(folder_path)
        if not image_paths:
            messagebox.showwarning("No images", "No supported image files were found in that folder.")
            return

        processed_dir = Path(folder_path).parent / "Processed_Images"
        self.prepare_output_folder(processed_dir)

        start = time.time()
        all_face_features = []
        all_saved_face_paths = []
        total_faces = 0
        last_input_image = None
        last_output_image = None

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            output_bgr, faces_info = self.process_single_image(image)
            last_input_image = image.copy()
            last_output_image = output_bgr.copy()

            for face_index, info in enumerate(faces_info, start=1):
                face_crop = info["crop"]
                feature = self.compute_face_feature(face_crop)
                all_face_features.append(feature)

                temp_name = f"temp_face_{len(all_saved_face_paths) + 1:03d}.jpg"
                save_path = processed_dir / temp_name
                cv2.imwrite(str(save_path), face_crop)
                all_saved_face_paths.append(save_path)
                total_faces += 1

        if last_input_image is not None and last_output_image is not None:
            self.current_input_bgr = last_input_image
            self.current_output_bgr = last_output_image
            self.show_on_canvas(self.input_canvas, self.current_input_bgr, "input")
            self.show_on_canvas(self.output_canvas, self.current_output_bgr, "output")

        if total_faces == 0:
            elapsed = time.time() - start
            self.status_var_1.set(f"Total {len(image_paths)} image(s) processed in {elapsed:.2f} seconds.")
            self.status_var_2.set("0 faces detected.")
            self.status_var_3.set(f"Processed images folder: {processed_dir}")
            return

        labels, unique_identities = self.cluster_faces(all_face_features)

        for i, temp_path in enumerate(all_saved_face_paths):
            identity_num = labels[i] + 1 if labels[i] >= 0 else unique_identities + (i + 1)
            new_name = f"Identity_{identity_num}_face_{i + 1}.jpg"
            new_path = temp_path.parent / new_name
            if new_path.exists():
                new_path.unlink()
            temp_path.rename(new_path)

        elapsed = time.time() - start
        self.status_var_1.set(f"Total {len(image_paths)} image(s) processed in {elapsed:.2f} seconds.")
        self.status_var_2.set(f"{total_faces} face(s) detected corresponding to {unique_identities} unique identit(ies).")
        self.status_var_3.set(f"Saved aligned crops to: {processed_dir}")

    def list_image_files(self, folder_path):
        supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = []
        for entry in sorted(Path(folder_path).iterdir()):
            if entry.is_file() and entry.suffix.lower() in supported:
                paths.append(entry)
        return paths

    def prepare_output_folder(self, output_dir):
        output_dir.mkdir(exist_ok=True)
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()

    # -------------------------------------------------------
    # FACE DETECTION
    # -------------------------------------------------------

    def detect_faces(self, image_bgr):
        h, w = image_bgr.shape[:2]

        # DNN SSD expects a 300x300 blob
        blob = cv2.dnn.blobFromImage(
            image_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False
        )
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.dnn_confidence_threshold:
                continue

            # DNN returns normalised coords, convert to pixel positions
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            # expand into a square box with some padding
            bw = x2 - x1
            bh = y2 - y1
            side = int(max(bw, bh) * 1.25)
            cx = x1 + bw // 2
            cy = y1 + bh // 2

            sx1 = max(0, cx - side // 2)
            sy1 = max(0, cy - side // 2)
            sx2 = min(w, sx1 + side)
            sy2 = min(h, sy1 + side)

            # keep it square after clipping
            side = min(sx2 - sx1, sy2 - sy1)
            sx2 = sx1 + side
            sy2 = sy1 + side

            if side > 0:
                boxes.append((sx1, sy1, sx2, sy2))

        boxes = self.filter_and_limit_boxes(image_bgr, boxes)
        return boxes

    def skin_mask_ratio(self, image_bgr, box):
        x1, y1, x2, y2 = box
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        hsv_mask = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([25, 180, 255]))
        ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
        skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)
        return skin_ratio

    def filter_and_limit_boxes(self, image_bgr, boxes):
        scored = []
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            skin_ratio = self.skin_mask_ratio(image_bgr, box)
            score = area * (0.5 + skin_ratio)
            if skin_ratio >= 0.05:
                scored.append((score, box))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in scored[:self.max_faces_per_image]]
        selected.sort(key=lambda b: (b[1], b[0]))
        return selected

    # -------------------------------------------------------
    # LANDMARK DETECTION (Component 3)
    # -------------------------------------------------------

    def detect_landmarks(self, image_bgr, box):
        """
        Run mediapipe face landmarker on the cropped face region and return
        the three landmark positions in full-image coordinates.
        Returns (right_eye, left_eye, nose_tip) as pixel coords,
        or None if detection fails.
        """
        x1, y1, x2, y2 = box
        face_crop = image_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        # wrap the numpy array as a mediapipe Image for the Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        face_lm = results.face_landmarks[0]
        crop_h, crop_w = face_crop.shape[:2]

        # pull out the three landmarks we need
        # Tasks API gives normalised 0-1 coords via .x and .y attributes
        r_eye = face_lm[self.RIGHT_EYE_IDX]
        l_eye = face_lm[self.LEFT_EYE_IDX]
        nose = face_lm[self.NOSE_TIP_IDX]

        # convert to pixel coords and offset by box origin for full-image coords
        right_eye = np.array([r_eye.x * crop_w + x1,
                              r_eye.y * crop_h + y1], dtype=np.float32)
        left_eye = np.array([l_eye.x * crop_w + x1,
                             l_eye.y * crop_h + y1], dtype=np.float32)
        nose_tip = np.array([nose.x * crop_w + x1,
                             nose.y * crop_h + y1], dtype=np.float32)

        return right_eye, left_eye, nose_tip

    # -------------------------------------------------------
    # FACIAL ALIGNMENT (Component 4)
    # -------------------------------------------------------

    def align_face(self, image_bgr, landmarks):
        """
        Compute a similarity transform that maps the 3 detected landmarks
        to the target positions in a 125x125 image, then warp.
        Returns the 125x125 aligned face crop.
        """
        right_eye, left_eye, nose_tip = landmarks

        src_pts = np.array([right_eye, left_eye, nose_tip], dtype=np.float32)
        dst_pts = self.target_landmarks.copy()

        # estimateAffinePartial2D fits a similarity transform (4 DOF: rotation, uniform scale, tx, ty)
        # using least squares on our 3 point pairs
        transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if transform_matrix is None:
            # fallback: just crop and resize if the transform fails
            return cv2.resize(image_bgr, (self.aligned_size, self.aligned_size))

        # warp the FULL image so the face lands at the target positions
        # output size is 125x125, so align + crop + resize all happen in one step
        aligned = cv2.warpAffine(
            image_bgr,
            transform_matrix,
            (self.aligned_size, self.aligned_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned

    def draw_landmark_circles(self, image, right_eye, left_eye, nose_tip, radius=4):
        """
        Draw coloured circles at landmark positions.
        Red = right eye, Green = left eye, Blue = nose tip.
        Coordinates should be integer pixel positions in the given image.
        """
        re = (int(round(right_eye[0])), int(round(right_eye[1])))
        le = (int(round(left_eye[0])), int(round(left_eye[1])))
        nt = (int(round(nose_tip[0])), int(round(nose_tip[1])))

        cv2.circle(image, re, radius, (0, 0, 255), -1)    # red
        cv2.circle(image, le, radius, (0, 255, 0), -1)     # green
        cv2.circle(image, nt, radius, (255, 0, 0), -1)     # blue

    # -------------------------------------------------------
    # SINGLE IMAGE PROCESSING (ties everything together)
    # -------------------------------------------------------

    def process_single_image(self, image_bgr):
        output = image_bgr.copy()
        h, w = output.shape[:2]
        boxes = self.detect_faces(image_bgr)
        faces_info = []

        # the four corner positions where aligned faces will be placed
        corners = [
            (0, 0),                                          # top-left
            (w - self.aligned_size, 0),                      # top-right
            (0, h - self.aligned_size),                      # bottom-left
            (w - self.aligned_size, h - self.aligned_size)   # bottom-right
        ]

        corner_idx = 0  # tracks which corner to use next

        for box in boxes:
            x1, y1, x2, y2 = box

            # detect landmarks first - if they fail, this isn't a real face
            landmarks = self.detect_landmarks(image_bgr, box)
            if landmarks is None:
                continue

            right_eye, left_eye, nose_tip = landmarks

            # landmarks found so this is a valid face - draw the bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2),
                          self.face_box_color, self.face_box_thickness)

            # draw landmark circles on the main output image
            self.draw_landmark_circles(output, right_eye, left_eye, nose_tip, radius=4)

            # compute the aligned 125x125 crop
            aligned_crop = self.align_face(image_bgr, landmarks)

            # draw landmark circles on the aligned crop at target positions
            self.draw_landmark_circles(
                aligned_crop,
                self.target_landmarks[0],  # right eye target
                self.target_landmarks[1],  # left eye target
                self.target_landmarks[2],  # nose target
                radius=3
            )

            # place the aligned crop in the next available corner
            if corner_idx < len(corners):
                cx, cy = corners[corner_idx]
                output[cy:cy + self.aligned_size, cx:cx + self.aligned_size] = aligned_crop
                corner_idx += 1

            # for bulk processing, save a clean crop without landmarks
            clean_crop = self.align_face(image_bgr, landmarks)

            faces_info.append({
                "box": box,
                "crop": clean_crop,
                "landmarks": landmarks
            })

        return output, faces_info

    # -------------------------------------------------------
    # CLUSTERING (Component 5)
    # -------------------------------------------------------

    def compute_face_feature(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (48, 48))
        feature = resized.astype(np.float32).flatten() / 255.0
        return feature

    def cluster_faces(self, features):
        if len(features) == 1:
            return np.array([0]), 1

        feature_matrix = np.vstack(features)
        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        clustering = DBSCAN(eps=18.0, min_samples=1, metric="euclidean")
        labels = clustering.fit_predict(feature_matrix)

        unique_non_noise = sorted(set(label for label in labels if label != -1))
        label_map = {old: new for new, old in enumerate(unique_non_noise)}
        remapped = np.array([label_map.get(label, -1) for label in labels])

        unique_identities = len(set(remapped)) - (1 if -1 in remapped else 0)
        if unique_identities == 0 and len(remapped) > 0:
            unique_identities = len(remapped)

        return remapped, unique_identities


def main():
    root = tk.Tk()
    app = FaceProjectGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()