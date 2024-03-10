import os

# Specify the paths to your folders
images_folder_path = "../data/sign_vs_no/train/sign"
labels_folder_path = "../labels/sign"

# Get the base names (without extension) of all label files
label_files = {
    os.path.splitext(f)[0]
    for f in os.listdir(labels_folder_path)
    if os.path.isfile(os.path.join(labels_folder_path, f))
}

# Iterate over all files in the images folder
for image_file in os.listdir(images_folder_path):
    image_base_name = os.path.splitext(image_file)[0]

    # Check if there's a corresponding label file by comparing base names
    if image_base_name not in label_files:
        # No corresponding label file found, delete the image
        image_path = os.path.join(images_folder_path, image_file)
        os.remove(image_path)
        print(f"Deleted image: {image_file}")

print(
    "Finished processing. Images without corresponding label files have been deleted."
)
