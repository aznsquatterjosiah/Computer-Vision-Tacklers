
# ============================================
# Names: Josiah Villaflor
# Student Number: 23389559
# CITS4402 Project - GUI starter with face detection and bulk processing
# ============================================

import os
import time
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class FaceProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image GUI")
        self.root.configure(bg="#d9d9d9")

        self.box_w = 360
        self.box_h = 260
        self.max_faces_per_image = 4
        self.face_box_color = (255, 220, 120)  # light blue in BGR-ish appearance when drawn by OpenCV
        self.face_box_thickness = 2

        self.original_tk = None
        self.output_tk = None

        self.current_input_bgr = None
        self.current_output_bgr = None

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

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
        self.status_var_3.set(f"Saved aligned crops placeholder to: {processed_dir}")

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

    def detect_faces(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        squares = []
        for (x, y, w, h) in detections:
            side = int(max(w, h) * 1.25)
            cx = x + w // 2
            cy = y + h // 2

            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(image_bgr.shape[1], x1 + side)
            y2 = min(image_bgr.shape[0], y1 + side)

            # ensure square stays square after clipping
            side = min(x2 - x1, y2 - y1)
            x2 = x1 + side
            y2 = y1 + side

            if side > 0:
                squares.append((x1, y1, x2, y2))

        squares = self.filter_and_limit_boxes(image_bgr, squares)
        return squares

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

    def process_single_image(self, image_bgr):
        output = image_bgr.copy()
        boxes = self.detect_faces(image_bgr)
        faces_info = []

        for box in boxes:
            x1, y1, x2, y2 = box
            crop = image_bgr[y1:y2, x1:x2].copy()

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                self.face_box_color,
                self.face_box_thickness
            )

            faces_info.append({
                "box": box,
                "crop": crop
            })

        return output, faces_info

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
