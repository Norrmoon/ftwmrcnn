#%%########################################################################################
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
#%%########################################################################################
import rasterio
import numpy as np
from PIL import Image
from mrcnn.utils import Dataset
from skimage.io import imread
#%%########################################################################################
# GeoTIFF fájl beolvasása
def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()  # Tömbként olvassuk be
        # Csatornák (RGB + NIR) sorrendjének ellenőrzése
        if image.shape[0] == 4:  # RGB + NIR
            image = np.transpose(image, (1, 2, 0))  # Csatornák az utolsó dimenzióba
        # UInt16 adatok skálázása
        image = (image / 65535.0 * 255).astype(np.uint8)  # 0-255 tartomány
    return image

#%%########################################################################################
# Maszkok beolvasása
def create_binary_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)
    binary_mask = np.where(mask_array > 0, 1, 0)
    return binary_mask

#%%########################################################################################
class FieldDataset(Dataset):
    def load_fields(self, dataset_dir, subset):
        # 1. Osztály hozzáadása
        self.add_class("dataset", 1, "field")

        # 2. Adatútvonalak beállítása
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_dir = os.path.join(dataset_dir, "images")
        mask_dir = os.path.join(dataset_dir, "masks")

        # 3. Képek és maszkok betöltése
        for filename in os.listdir(image_dir):
            image_id = os.path.splitext(filename)[0]
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            # Ellenőrizzük, hogy a maszk létezik-e
            if not os.path.exists(mask_path):
                continue

            self.add_image(
                "dataset",
                image_id=image_id,
                path=img_path,
                mask_path=mask_path
            )

    def load_image(self, image_id):
        # GeoTIFF kép betöltése rasterio-val
        info = self.image_info[image_id]
        image = load_geotiff(info['path'])
        return image[..., :3]  # Csak az RGB csatornákat tartjuk meg

    def load_mask(self, image_id):
        # GeoTIFF maszk betöltése rasterio-val
        info = self.image_info[image_id]
        mask = load_geotiff(info['mask_path'])
        # Bináris maszk készítése (0 = háttér, 1 = érdekes terület)
        mask = np.where(mask > 0, 1, 0).astype(np.bool_)
        # Győződj meg róla, hogy a maszk formátuma (256, 256, 1)
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)  # Eltávolítja az első dimenziót
        # Extra dimenzió hozzáadása az osztály dimenzióhoz
        mask = mask[..., np.newaxis]
        class_ids = np.array([1])  # Csak egy osztály van
        # Ellenőrzés, hogy a maszk tartalmaz-e legalább egy "1" értéket
        if mask.sum() == 0:
            print(f"Skipping image_id={image_id} due to empty mask.")
            return None, None  # Címkézetlen adatként visszatér
        return mask, class_ids

#%%########################################################################################
# Adatbetöltő példányok létrehozása
dataset_train = FieldDataset()
dataset_train.load_fields('dataset', 'train')
dataset_train.prepare()

dataset_val = FieldDataset()
dataset_val.load_fields('dataset', 'val')
dataset_val.prepare()

#%%########################################################################################

# Konfigurációs osztály létrehozása
from mrcnn.config import Config

class FieldConfig(Config):
    NAME = "field_segmentation"
    NUM_CLASSES = 1 + 1  # háttér + mező
    STEPS_PER_EPOCH = 375
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    BATCH_SIZE = 1
    IMAGES_PER_GPU = 1
    RUN_EAGERLY = False
    IMAGE_CHANNEL_COUNT = 3
    DETECTION_MAX_INSTANCES = 50
    # További konfigurációk, mint BATCH_SIZE, IMAGE_MIN_DIM, IMAGE_MAX_DIM, stb.
#%%########################################################################################
from mrcnn.model import MaskRCNN

config = FieldConfig()
model = MaskRCNN(mode='training', config=config, model_dir='./logs')

# Előre betanított súlyok betöltése (opcionális, ha van)
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
#     "mrcnn_class_logits", "mrcnn_bbox_fc",
#     "mrcnn_bbox", "mrcnn_mask"])
#for image_id in dataset_train.image_ids[:1]:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    print(f"Image shape: {image.shape}")
#    print(f"Mask shape: {mask.shape}")
#    print(f"Class IDs: {class_ids}")
# Tanítás
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='all')  # vagy 'heads', ha csak a fejrétegeket akarod tanítani
#%%########################################################################################
# Modell mentése
model_path = 'field_segmentation.h5'
model.keras_model.save_weights(model_path)
#%%########################################################################################
# Betöltés inferenciához
inference_config = FieldConfig()
inference_config.GPU_COUNT = 1
inference_config.IMAGES_PER_GPU = 1

model = MaskRCNN(mode='inference', config=inference_config, model_dir='./logs')
model.load_weights(model_path, by_name=True)
#%%########################################################################################
# Tesztelés és előrejelzés: Kép betöltése
image = imread('path/to/test/image.png')
#%%########################################################################################
# Előrejelzés
results = model.detect([image], verbose=1)
r = results[0]
#%%########################################################################################
# Eredmények megjelenítése
from mrcnn import visualize

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            ['BG', 'field'], r['scores'])
#%%########################################################################################
# példa eredmények értékelésére
from mrcnn.utils import compute_ap

APs = []
for image_id in dataset_val.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        mrcnn.model.load_image_gt(dataset_val, inference_config, image_id)
    results = model.detect([image], verbose=0)
    r = results[0]
    AP, precisions, recalls, overlaps = \
        compute_ap(gt_bbox, gt_class_id, gt_mask,
                   r['rois'], r['class_ids'], r['scores'], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))
