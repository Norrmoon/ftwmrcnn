#%%####################################################################
import os
import numpy as np
import rasterio
import shutil
#%%####################################################################
# Mappa elérési utak
original_images_dir = "original_images"
original_masks_dir = "original_masks"
valid_images_dir = "valid_images"
invalid_images_dir = "invalid_images"
valid_masks_dir = "valid_masks"
invalid_masks_dir = "invalid_masks"

# Létrehozza a célmappákat, ha nem léteznek
for directory in [valid_images_dir, invalid_images_dir, valid_masks_dir, invalid_masks_dir]:
    os.makedirs(directory, exist_ok=True)

#%%####################################################################
# Inicializálja a tömböket
valid_files = []
invalid_files = []

# Maszkok ellenőrzése
for mask_file in os.listdir(original_masks_dir):
    mask_path = os.path.join(original_masks_dir, mask_file)
    
    # Ellenőrizze, hogy a fájl .tif kiterjesztésű-e
    if mask_file.endswith(".tif"):
        with rasterio.open(mask_path) as mask:
            mask_data = mask.read(1)  # Az első sáv olvasása
            if np.all(mask_data == 0):  # Ha minden érték 0
                invalid_files.append(mask_file)
            else:
                valid_files.append(mask_file)
#%%####################################################################
# Fájlok átmásolása a megfelelő mappákba
for file_list, source_dir, target_dir_images, target_dir_masks in [
    (valid_files, original_images_dir, valid_images_dir, valid_masks_dir),
    (invalid_files, original_images_dir, invalid_images_dir, invalid_masks_dir),
]:
    for file_name in file_list:
        image_file = os.path.join(source_dir, file_name)
        mask_file = os.path.join(original_masks_dir, file_name)

        # Másolás a célmappákba
        shutil.copy(image_file, os.path.join(target_dir_images, file_name))
        shutil.copy(mask_file, os.path.join(target_dir_masks, file_name))

print(f"Érvényes fájlok: {len(valid_files)}")
print(f"Érvénytelen fájlok: {len(invalid_files)}")

#%%############################################################################################
###############################################################################################
import random

# Mappa elérési utak a dataset struktúrához
train_images_dir = "dataset/train/images"
train_masks_dir = "dataset/train/masks"
val_images_dir = "dataset/val/images"
val_masks_dir = "dataset/val/masks"
test_images_dir = "dataset_test/images"
test_masks_dir = "dataset_test/masks"

# Létrehozza a célmappákat, ha nem léteznek
for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, test_images_dir, test_masks_dir]:
    os.makedirs(directory, exist_ok=True)
#%%####################################################################
# Kevert valid_files lista
random.shuffle(valid_files)

# Első 3000 fájl -> train
train_files = valid_files[:3000]
# Következő 300 fájl -> val
val_files = valid_files[3000:3300]
# Következő 300 fájl -> test
test_files = valid_files[3300:3600]

# Másolási függvény
def copy_files(file_list, source_images_dir, source_masks_dir, target_images_dir, target_masks_dir):
    for file_name in file_list:
        image_path = os.path.join(source_images_dir, file_name)
        mask_path = os.path.join(source_masks_dir, file_name)

        shutil.copy(image_path, os.path.join(target_images_dir, file_name))
        shutil.copy(mask_path, os.path.join(target_masks_dir, file_name))
#%%####################################################################
# Másolás a train mappákba
copy_files(train_files, valid_images_dir, valid_masks_dir, train_images_dir, train_masks_dir)

# Másolás a val mappákba
copy_files(val_files, valid_images_dir, valid_masks_dir, val_images_dir, val_masks_dir)

# Másolás a test mappákba
copy_files(test_files, valid_images_dir, valid_masks_dir, test_images_dir, test_masks_dir)

print(f"Train fájlok: {len(train_files)}")
print(f"Validation fájlok: {len(val_files)}")
print(f"Test fájlok: {len(test_files)}")
# %%
