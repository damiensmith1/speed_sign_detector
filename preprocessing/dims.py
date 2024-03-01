import os
from PIL import Image


def print_image_dimensions(folder_path):
    # Iterate over the files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Check if the file is a JPG image
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    print(f"{filename}: {width}x{height}")
            except IOError:
                print(f"Error opening or reading image {filename}")


# Replace 'path/to/your/folder' with the actual path to the folder containing the JPG images
folder_path = "../data/train/nosignresized"
print_image_dimensions(folder_path)
