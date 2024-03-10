import ast
import re


# read text file

labelspth = "../labels/test_labels.txt"
processed_files = set()
with open(labelspth, "r") as file:
    labels = file.read()

lines = labels.strip().split("\n")
print(len(lines))
width = 300
height = 300

for line in lines:
    pattern = r'([\w\.]+),"\((\d+), (\d+)\)","\((\d+), (\d+)\)"'

    # Use re.search to find matches
    match = re.search(pattern, line)

    if match:
        file_name = match.group(1)
        x_min, y_min = int(match.group(2)), int(match.group(3))
        x_max, y_max = int(match.group(4)), int(match.group(5))

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height

    w = (x_max - x_min) / width
    h = (y_max - y_min) / height

    x_center = round(x_center, 6)
    y_center = round(y_center, 6)
    w = round(w, 6)
    h = round(h, 6)

    labelfile = "../labels/test/sign/" + file_name.replace(".jpg", ".txt")

    if file_name in processed_files:
        print(file_name)

    # Check if file has been processed; if not, clear it (to avoid duplicating from previous runs)
    if file_name not in processed_files:
        processed_files.add(file_name)

    with open(labelfile, "w") as lfile:
        lfile.write(f"0 {x_center} {y_center} {w} {h}\n")

    # After processing all lines
print("Number of processed files:", len(processed_files))
