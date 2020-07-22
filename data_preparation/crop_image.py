from PIL import Image
import os
import random
from pathlib import Path
from shutil import copyfile
import pandas as pd
import cv2

def random_crop(data_path, new_data_path):
    # Define crop info
    crop_number = 5
    crop_tl_size = 15
    crop_br_size = 35

    # Dir info
    list_dir = os.listdir(data_path)
    total = len(list_dir)

    for index, img_name in enumerate(list_dir):
        if img_name.endswith('.png'):
            print("Processing image: ", img_name, '\nNumber: ', index, '/', total)
            img_path = os.path.join(data_path, img_name)
            
            # Load image
            img = Image.open(img_path)
            x, y = img.size
            
            # Img dir and path
            new_crop_dir = img_name.replace('.png', '')
            Path(os.path.join(new_data_path, new_crop_dir)).mkdir(parents=True, exist_ok=True)
            copyfile(img_path, os.path.join(new_data_path, new_crop_dir, img_name))

            # Random crop image
            for i in range(crop_number):
                # Make random range
                x_tl = random.randrange(0, crop_tl_size)
                y_tl = random.randrange(0, crop_tl_size)
                x_br = random.randrange(crop_br_size, x)
                y_br = random.randrange(crop_br_size, y)

                # Crop then resize image
                crop_img_name = 'Crop-tl-(' + str(x_tl) + ',' + str(y_tl) + ')-Crop-br-(' + str(x_br) + ',' + str(y_br) + ')-' + img_name
                crop_img = img.crop((x_tl, y_tl, x_br, y_br))
                crop_img = crop_img.resize((x, y))
                crop_img.save(os.path.join(new_data_path, new_crop_dir, crop_img_name))

def crop_by_radius(img, x, y, x_radius, y_radius):
    xmin = (le[0] - x_radius) if (le[0] - x_radius) > 0 else 0
    ymin = (le[1] - y_radius) if (le[1] - y_radius) > 0 else 0
    xmax = (le[0] + x_radius) if (le[0] + x_radius) < x else x - 1
    ymax = (le[1] + y_radius) if (le_ymin + y_radius) < y else y - 1 
    new_img = img[le_ymin:le_ymax, le_xmin:le_xmax]
    return img

def mtcnn_crop(data_path, new_data_path, radius=0.4):
    from mtcnn import MTCNN
    detector = MTCNN()
    # Define crop info
    crop_number = 5

    # Dir info
    list_dir = os.listdir(data_path)
    total = len(list_dir)

    for index, img_name in enumerate(list_dir):
        if img_name.endswith('.png'):
            print("Processing image: ", img_name, '\nNumber: ', index, '/', total)
            img_path = os.path.join(data_path, img_name)
            
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect 5 facial landmark
            img_det = detector.detect_faces(img)
            facial_landmarks = img_det[0]['keypoints']            
            # le = list(facial_landmarks['left_eye'])
            # re = list(facial_landmarks['right_eye'])
            # nose = list(facial_landmarks['nose'])
            # ml = list(facial_landmarks['mouth_left'])
            # mr = list(facial_landmarks['mouth_right'])

            # Calc size crop
            x, y = img.shape[0], img.shape[1]
            x_radius = x*radius
            y_radius = y*radius
            
            for index, values in enumerate(facial_landmarks):
                new_img = crop_by_radius(img, x, y, x_radius, y_radius)
                cv2.imwrite(index + )



def fixed_crop(data_path, new_data_path):


def create_label_file(origin_path, data_path, label_file_name, crop_number=6):
    fer_label_csv = os.path.join(origin_path, "label.csv")
    
    label_file_txt = os.path.join(data_path, 'label', label_file_name + "label.txt")
    list_file_txt = os.path.join(data_path, 'label', label_file_name + "list.txt")

    cols = pd.read_csv(fer_label_csv, nrows=1).columns
    cols = cols.delete(1)
    fer = pd.read_csv(fer_label_csv, usecols=cols)

    fer_data = fer.drop('Name', axis=1)
    fer_most_voting = fer_data.idxmax(axis=1)
    fer_most_voting_values = fer_data.max(axis=1)
    index, fer_most_voting_label = zip(*fer_most_voting.items())
    data_mapping = {'img_name': [], 'frame': []}
    label_list = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    with open(label_file_txt, 'w+') as label_file_txt:
        for id in range(len(fer_most_voting_label)):
            label = fer_most_voting_label[id]
            if (fer_most_voting_values[id] >=5) and (label not in ['NF', 'contempt', 'unknown']):
                label_index = label_list.index(label)
                label_file_txt.write(str(label_index)+'\n')
                img_name = fer['Name'][id]
                data_mapping['img_name'].append(str(label_index) + '/'+ str(img_name))
                data_mapping['frame'].append(crop_number)
    
    data_mapping_pd = pd.DataFrame.from_dict(data_mapping)
    data_mapping_pd.to_csv(list_file_txt, sep=' ', header= False, index=False)


def copy_org_img(test_path, new_data_path):
    list_dir = os.listdir(test_path)
    for index, img_name in enumerate(list_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(test_path, img_name)
            new_crop_dir = img_name.replace('.png', '')
            path = Path(os.path.join(new_data_path, new_crop_dir)).mkdir(parents=True, exist_ok=True)
            copyfile(img_path, os.path.join(new_data_path, new_crop_dir, img_name))


def main(data_path, new_data_path, label_file_name):
    # create_label_file(origin_path, new_data_path, label_file_name, crop_number=6)

    # # Choose 1 of 3 cropping methods
    # random_crop(data_path, new_data_path)
    mtcnn_crop(data_path, new_data_path)
    # fixed_crop(data_path, new_data_path)


    # copy_org_img(test_path, new_data_path)


if __name__ == '__main__':
    origin_path = '/data/ngocnkd/ngocnkd/FER_dataset/FER2013Valid'
    new_path = '/data/ngocnkd/ngocnkd/region-attention-network/New_Data/FER_valid'
    label_file_name = "ferplus_random_crop_val_"
    main(origin_path, new_path, label_file_name)
