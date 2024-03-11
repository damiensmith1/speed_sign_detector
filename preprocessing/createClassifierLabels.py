import os

image_folder_path = "../data/onlysigns/test/100"

output_file_path = "../data/onlysigns/test/labels.txt"
image_files = os.listdir(image_folder_path)

with open(output_file_path, "a") as file:
    for image_name in image_files:
        file.write(image_name + "," + "100" + "\n")

print(f"Image names have been written to {output_file_path}")
