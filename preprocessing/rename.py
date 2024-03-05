import os
from PIL import Image
import time


def convert_and_rename_images(src_folder, dest_folder):
    if not src_folder.endswith("/"):
        src_folder += "/"
    if not dest_folder.endswith("/"):
        dest_folder += "/"

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = os.listdir(src_folder)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for file_name in image_files:
        src_file_path = os.path.join(src_folder, file_name)

        with Image.open(src_file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            current_time_milliseconds = int(time.time() * 1000)
            new_file_name = f"{current_time_milliseconds}.jpg"
            new_file_path = os.path.join(dest_folder, new_file_name)

            img.save(new_file_path, "JPEG")


src_folder = r"D:\Downloads\Data-20240216T210519Z-001\Data\signs\signs\\110"
dest_folder = r"D:\Downloads\Data-20240216T210519Z-001\Data\signs\newdata\sign"
convert_and_rename_images(src_folder, dest_folder)
