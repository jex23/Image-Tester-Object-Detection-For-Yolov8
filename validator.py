import os
import cv2
import cvzone
import math
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO


class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO App")

        self.model_path = "../Yolo-Weights/best.pt"
        self.model = YOLO(self.model_path)
        self.class_names = ["balut", "penoy"]

        self.image_paths = []
        self.current_image_index = 0

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.next_button = ttk.Button(root, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.prev_button = ttk.Button(root, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side=tk.RIGHT)

        self.select_folder_button = ttk.Button(root, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(side=tk.LEFT)

        self.load_images()
        self.display_image()

    def load_images(self):
        folder_path = "../Yolo-Weights"  # Replace with your default folder path
        self.image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                            file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths.sort()

    def display_image(self):
        if not self.image_paths:
            return

        image_path = self.image_paths[self.current_image_index]
        self.image = cv2.imread(image_path)
        self.results = self.model(self.image)

        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        self.canvas.config(width=self.img_tk.width(), height=self.img_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        for r in self.results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Display bounding box and text on canvas
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FF00FF", width=3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label_text = f'{self.class_names[cls]} {conf}'
                self.canvas.create_text(max(0, x1), max(35, y1), anchor=tk.NW, text=label_text, fill="#FFFFFF")

    def show_next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_image()

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image()

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                                file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths.sort()
            self.current_image_index = 0
            self.display_image()


def main():
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
