#!/bin/bash

source_directory="${0%/*}"

dest_dir1="${source_directory}/split_dir1"
dest_dir2="${source_directory}/split_dir2"
dest_dir3="${source_directory}/split_dir3"
dest_dir4="${source_directory}/split_dir4"

mkdir -p "${dest_dir1}" "${dest_dir2}" "${dest_dir3}" "${dest_dir4}"

total_images=$(find "${source_directory}" -type f -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" | wc -l)

images_per_dir=$((total_images / 4))

i=0
find "${source_directory}" -type f -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" | while read -r file; do
    if ((i < images_per_dir)); then
        echo "Moving $file to $dest_dir1"
        mv "$file" "$dest_dir1"
    elif ((i < 2*images_per_dir)); then
        echo "Moving $file to $dest_dir2"
        mv "$file" "$dest_dir2"
    elif ((i < 3*images_per_dir)); then
        echo "Moving $file to $dest_dir3"
        mv "$file" "$dest_dir3"
    else
        echo "Moving $file to $dest_dir4"
        mv "$file" "$dest_dir4"
    fi
    ((i++))
done

echo "Done splitting"

