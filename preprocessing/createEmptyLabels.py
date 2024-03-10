import os

images_folder_path = "../data/sign_vs_no/test/nosign"
labels_folder_path = "../labels/test/nosign"

image_files = [
    f
    for f in os.listdir(images_folder_path)
    if os.path.isfile(os.path.join(images_folder_path, f))
]

for image_file in image_files:
    label_file_name = os.path.splitext(image_file)[0] + ".txt"
    label_file_path = os.path.join(labels_folder_path, label_file_name)
    open(label_file_path, "w").close()
