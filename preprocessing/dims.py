import os
from PIL import Image


def print_image_dimensions(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    print(f"{filename}: {width}x{height}")
            except IOError:
                print(f"Error opening or reading image {filename}")


folder_path = "../data/onlysigns/train/40croppedaugmented"
print_image_dimensions(folder_path)
