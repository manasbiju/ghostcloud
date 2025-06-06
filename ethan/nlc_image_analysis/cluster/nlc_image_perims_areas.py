"""
Created May 22 2025
Updated Jun 01 2025

(IN CLUSTER)
Compute the perimeter & area for each cloud in some NLC image

Saves the perimeters & areas in one .npy file for each threshold in which there's at least one cloud that doesn't touch
the boundary
"""
import sys
import os
import re
import numpy as np
from clouds_helpers import label_image, fill_and_label_image, set_thread_count, get_perimeters_areas

# ============================
# Section: Import command-line arguments
# ============================

id_num = int(sys.argv[1])
thread_count = int(sys.argv[2])
set_thread_count(thread_count)

# ============================
# Section: Set parameters & file paths
# ============================

# Get the current PNG file's location
png_directory = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images/xsc1/good'  
# These are the PNG file names WITHOUT parent directory path
png_file_names = [file for file in os.listdir(png_directory) if file.endswith('.png')]  
png_file_name_with_id = next((name for name in png_file_names if re.search(f'id={id_num}.png$', name)), None)

# ============================
# Section: Fill holes & remove border clouds
# ============================

check_dir = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_areas_perims/fill'
# Names of the folders in check_dir, i.e., one will be 2012-12-30--03-56-29--421_id=1_pa_fill
# Inside this folder I will be storing the perimeter-area .npy files for each threshold
save_dirs = [d for d in os.listdir(check_dir) if os.path.isdir(os.path.join(check_dir, d))]
temp_dir = png_file_name_with_id.split('.')[0] + f'_pa_fill'
save_dir = f'{check_dir}/{temp_dir}'
if temp_dir not in save_dirs:
    os.makedirs(name=save_dir)

for thresh in np.linspace(start=1, stop=255, num=255).astype(int):
    save_file_name = png_file_name_with_id.split('.')[0] + f'_pa_fill_thresh={thresh}.npy'
    processed_arr, num_features = fill_and_label_image(path=f'{png_directory}/{png_file_name_with_id}', thresh=thresh, rem_border_clouds=True)
    occup_prob = np.sum(processed_arr > 0) / processed_arr.size
    if occup_prob == 1.0 or occup_prob == 0.0:  
        continue
    else:
        perims, areas = get_perimeters_areas(arr=processed_arr)
        np.save(f'{save_dir}/{save_file_name}', np.array([perims, areas]))

# ============================
# Section: Don't fill holes & remove border clouds
# ============================

check_dir = '/projects/illinois/eng/physics/dahmen/mullen/Clouds/nlc_images_areas_perims/no_fill'
save_dirs = [d for d in os.listdir(check_dir) if os.path.isdir(os.path.join(check_dir, d))]
temp_dir = png_file_name_with_id.split('.')[0] + f'_pa_nofill'
save_dir = f'{check_dir}/{temp_dir}'
if temp_dir not in save_dirs:
    os.makedirs(name=save_dir)

for thresh in np.linspace(start=1, stop=255, num=255).astype(int):
    save_file_name = png_file_name_with_id.split('.')[0] + f'_pa_nofill_thresh={thresh}.npy'
    processed_arr, num_features = label_image(path=f'{png_directory}/{png_file_name_with_id}', thresh=thresh, rem_border_clouds=True)
    occup_prob = np.sum(processed_arr > 0) / processed_arr.size
    if occup_prob == 1.0 or occup_prob == 0.0:  
        continue
    else:
        perims, areas = get_perimeters_areas(arr=processed_arr)
        np.save(f'{save_dir}/{save_file_name}', np.array([perims, areas]))
