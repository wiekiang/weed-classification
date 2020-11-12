import os
import shutil
import requests
from zipfile import ZipFile
import pandas as pd

# Global variables
URL = "https://docs.google.com/uc?export=download"
CHUNK_SIZE = 32768

IMG_DIR = "./dataset/"
TRAIN_DIR = "./dataset/train/"
TEST_DIR = "./dataset/test/"
IMG_GD_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
IMG_ZIP_FILE = "./dataset/images.zip"

CLASS = 9
CLASS_FOLDER = ['chineeapple',
                'lantana',
                'parkinsonia',
                'parthenium',
                'pricklyacacia',
                'rubbervine',
                'siamweed',
                'snakeweed',
                'negatives']
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'test'


def download_google_drive_file(id, destination):
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def download_images():
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
        os.makedirs(TRAIN_DIR)
        os.makedirs(TEST_DIR)

        for i in range(0, CLASS):
            os.makedirs(TRAIN_DIR + '/' + CLASS_FOLDER[i])
            os.makedirs(TEST_DIR + '/' + CLASS_FOLDER[i])

        print("Downloading DeepWeeds images to " + IMG_ZIP_FILE)
        download_google_drive_file(IMG_GD_ID, IMG_ZIP_FILE)
        print("Finished downloading images.")
        print("Unzipping " + IMG_ZIP_FILE)
        with ZipFile(IMG_ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(IMG_DIR)
        print("Finished unzipping images.")
        os.remove(IMG_ZIP_FILE)


def train_dataset_processing():
    df = pd.read_csv('labels/labels.csv')
    print('Train dataset cleaning up....\n')

    for i in range(0, CLASS):
        total_images = len(df[df.Label == i])
        row = df.Filename[df.Label == i]
        class_name = CLASS_FOLDER[i]

        print(class_name + ': ' + str(total_images) + ' images')

        for image_name in row:
            source = IMG_DIR + image_name
            dest = TRAIN_DIR + class_name
            shutil.move(source, dest)


def test_dataset_processing():
    df = pd.read_csv('labels/labels.csv')
    print('\nGenerating test dataset\n')

    for i in range(0, CLASS):
        total_images = len(df[df.Label == i])
        test_images = int(total_images * .3)
        random_row = df.Filename[df.Label == i].sample(test_images)
        class_name = CLASS_FOLDER[i]

        print(class_name + ': ' + str(test_images) + ' of ' + str(total_images) + ' images')

        for image_name in random_row:
            source = TRAIN_DIR + class_name + '/' + image_name
            dest = TEST_DIR + class_name
            shutil.move(source, dest)


if __name__ == '__main__':
    download_images()
    train_dataset_processing()
    test_dataset_processing()
