import gdown
import os
import zipfile

TRAIN_DATA_PATH ='https://drive.google.com/file/d/14BidJeG4nSNuFNFDf99K-7eErCq4i47t/view?usp=sharing'
TEST_DATA_PATH = 'https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view?usp=sharing'
OUTPUT_FOLDER = 'data/'

if __name__ == "__main__":
    gdown.download(TRAIN_DATA_PATH, OUTPUT_FOLDER, quiet=False, fuzzy=True)
    gdown.download(TEST_DATA_PATH, OUTPUT_FOLDER, quiet=False, fuzzy=True)
    train_zip_path = os.path.join(OUTPUT_FOLDER, "train.zip")
    test_zip_path = os.path.join(OUTPUT_FOLDER, "test.zip")

    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_FOLDER)

    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall(OUTPUT_FOLDER)
        
    os.remove(train_zip_path)
    os.remove(test_zip_path)