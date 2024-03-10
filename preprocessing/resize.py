import os
from PIL import Image


def resize_and_pad_image(
    input_folder, output_folder, target_size=(300, 300), padding_color=(0, 0, 0)
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            original_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(original_path) as img:
                scale = min(target_size[0] / img.width, target_size[1] / img.height)
                new_size = (int(img.width * scale), int(img.height * scale))

                img = img.resize(new_size, Image.ANTIALIAS)

                new_img = Image.new("RGB", target_size, padding_color)
                upper_left = (
                    (target_size[0] - new_size[0]) // 2,
                    (target_size[1] - new_size[1]) // 2,
                )
                new_img.paste(img, upper_left)

                new_img.save(output_path)
                print(f"Processed {filename}")


input_folder = "../data/sign"
output_folder = "../data/sign-resized"
resize_and_pad_image(input_folder, output_folder)
