import os
from PIL import Image


def resize_and_pad_image(
    input_folder, output_folder, target_size=(300, 300), padding_color=(0, 0, 0)
):
    """
    Resize and pad images to a target size.

    Parameters:
    - input_folder: Path to the folder containing the original images.
    - output_folder: Path where the processed images will be saved.
    - target_size: A tuple (width, height) representing the target dimensions.
    - padding_color: A tuple (R, G, B) representing the color of the padding.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            original_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(original_path) as img:
                # Calculate the scaling factor to resize the image
                scale = min(target_size[0] / img.width, target_size[1] / img.height)
                new_size = (int(img.width * scale), int(img.height * scale))

                # Resize the image
                img = img.resize(new_size, Image.ANTIALIAS)

                # Create a new image with the specified background color and the target size
                new_img = Image.new("RGB", target_size, padding_color)
                # Calculate the position to paste the resized image
                upper_left = (
                    (target_size[0] - new_size[0]) // 2,
                    (target_size[1] - new_size[1]) // 2,
                )
                new_img.paste(img, upper_left)

                # Save the resized and padded image
                new_img.save(output_path)
                print(f"Processed {filename}")


input_folder = "../data/train/nosignraw"
output_folder = "../data/train/nosignresized"
resize_and_pad_image(input_folder, output_folder)
