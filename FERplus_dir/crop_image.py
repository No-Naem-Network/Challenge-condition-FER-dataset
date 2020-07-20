from PIL import Image
import os
import random
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile


def random_crop(data_path, new_data_path):
    f = []
    crop_number = 5
    crop_tl_size = 15
    crop_br_size = 35
    list_dir = os.listdir(test_path)
    total = len(list_dir)
    for index, img_name in tqdm(enumerate(list_dir)):
        if img_name.endswith('.png'):
            print("Processing image: ", img_name, '\nNumber: ', index, '/', total)
            img_path = os.path.join(test_path, img_name)
            img = Image.open(img_path)
            x, y = img.size
            new_crop_dir = img_name.replace('.png', '')
            Path(os.path.join(new_data_path, new_crop_dir)).mkdir(parents=True, exist_ok=True)
            for i in range(crop_number):
                x_tl = random.randrange(0, crop_tl_size)
                y_tl = random.randrange(0, crop_tl_size)
                x_br = random.randrange(crop_br_size, x)
                y_br = random.randrange(crop_br_size, y)
                crop_img_name = 'Crop-tl-(' + str(x_tl) + ',' + str(y_tl) + ')-Crop-br-(' + str(x_br) + ',' + str(y_br) + ')-' + img_name
                crop_img = img.crop((x_tl, y_tl, x_br, y_br))
                crop_img = crop_img.resize((x, y))
                crop_img.save(os.path.join(new_data_path, new_crop_dir, crop_img_name))

def copy_org_img(test_path, new_data_path):
    list_dir = os.listdir(test_path)
    for index, img_name in enumerate(list_dir):
        img_path = os.path.join(test_path, img_name)
        new_crop_dir = img_name.replace('.png', '')
        path = Path(os.path.join(new_data_path, new_crop_dir)).mkdir(parents=True, exist_ok=True)
        copyfile(img_path, os.path.join(new_data_path, new_crop_dir, img_name))

def main(test_path, new_data_path):
    random_crop(test_path, new_data_path)
    copy_org_img(test_path, new_data_path)



if __name__ == '__main__':
    origin_path = '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/Data/FER2013Test'
    new_path = '/home/oem/project/Face Expression/5. Challenge-condition-FER-dataset/New_Data/FER2013Test'

    main(origin_path, new_path)