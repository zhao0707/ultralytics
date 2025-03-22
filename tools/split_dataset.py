import os

from sklearn.model_selection import train_test_split


image_list = os.listdir('/home/pw/code/zsx/yolov11/ultralytics/datasets/images')
train_list, test_list = train_test_split(image_list, test_size=0.2, random_state=42)
val_list, test_list = train_test_split(test_list, test_size=0.5, random_state=42)

# print('total =', len(image_list))
# print('train :', len(train_list))
# print('val   :', len(val_list))
# print('test  :', len(test_list))

from pathlib import Path
from shutil import copyfile
from tqdm import tqdm


def copy_data(file_list, img_labels_root, imgs_source, mode):
    dataset_root = Path('/home/pw/code/zsx/yolov11/ultralytics/datasets')

    # Create directories if they don't exist
    images_path = dataset_root / 'images' / mode
    labels_path = dataset_root / 'labels' / mode
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    # Copying files with progress bar
    for file in tqdm(file_list, desc=f"Copying {mode} data"):
        base_filename = file.replace('.jpg', '')

        img_src_file = Path(imgs_source) / (base_filename + '.jpg')
        label_src_file = Path(img_labels_root) / (base_filename + '.txt')

        img_dest_file = images_path / (base_filename + '.jpg')
        label_dest_file = labels_path / (base_filename + '.txt')

        copyfile(img_src_file, img_dest_file)
        copyfile(label_src_file, label_dest_file)


# Example usage
copy_data(
    train_list,
    '/home/pw/code/zsx/yolov11/ultralytics/datasets/labels',
    '/home/pw/code/zsx/yolov11/ultralytics/datasets/images',
    "train")
copy_data(
    val_list,
    '/home/pw/code/zsx/yolov11/ultralytics/datasets/labels',
    '/home/pw/code/zsx/yolov11/ultralytics/datasets/images',
    "val")
copy_data(
    test_list,
    '/home/pw/code/zsx/yolov11/ultralytics/datasets/labels',
    '/home/pw/code/zsx/yolov11/ultralytics/datasets/images',
    "test")


