import os
from PIL import Image
import time


def convert_and_rename_images(src_folder, dest_folder):
    # Ensure both folder paths end with a slash
    if not src_folder.endswith("/"):
        src_folder += "/"
    if not dest_folder.endswith("/"):
        dest_folder += "/"

    # Create the destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all files in the source directory
    files = os.listdir(src_folder)

    # Filter for image files (by extension)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for file_name in image_files:
        # Create the full file path for the source image
        src_file_path = os.path.join(src_folder, file_name)

        # Open the image
        with Image.open(src_file_path) as img:
            # Ensure image is in RGB mode for JPG format
            if img.mode != "RGB":
                img = img.convert("RGB")
            current_time_milliseconds = int(time.time() * 1000)
            new_file_name = f"{current_time_milliseconds}.jpg"
            new_file_path = os.path.join(dest_folder, new_file_name)

            # Save the image in the destination folder as JPEG
            img.save(new_file_path, "JPEG")


# Example usage
src_folder = r"D:\Downloads\Data-20240216T210519Z-001\Data\signs\signs\\110"
dest_folder = r"D:\Downloads\Data-20240216T210519Z-001\Data\signs\newdata\sign"
convert_and_rename_images(src_folder, dest_folder)
