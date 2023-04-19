import argparse

class Settings():

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
        self.parser.add_argument("--dataset_path", type=str, required=True, help="path to training dataset")
        self.parser.add_argument("--style_image", type=str, default="styles/mosaic.jpg", help="path to style image")
        self.parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
        self.parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
        self.parser.add_argument("--image_size", type=int, default=256, help="Size of training images")
        self.parser.add_argument("--style_size", type=int, help="Size of style image")
        self.parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
        self.parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
        self.parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
        self.parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
        self.parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Batches between saving model")
        self.parser.add_argument("--sample_interval", type=int, default=1000, help="Batches between saving image samples")
        self.parser.add_argument("--result_to_tg", type=int, default=0, help="Need to send results in telegram")

    def parse_args(self):
        return self.parser.parse_args()


import os
from dotenv import load_dotenv
from pathlib import Path

class DotEnvControl():
    
    def __init__(self, config_file_path) -> None:

        load_env_result = load_dotenv(dotenv_path=config_file_path, verbose=True)
        print("load env ok" if load_env_result else "load env fail")

        self.PROJECT_DIR = os.environ.get("ROOT_DIR", "")
        self.ROOT_SRC_DIR = os.environ.get("ROOT_SRC_DIR", "/home/src")
        self.TG_NOTIFY_BOT_TOKEN = os.environ.get("TG_NOTIFY_BOT_TOKEN", "")
        self.TRAIN_STYLE_PATH = os.environ.get("TRAIN_STYLE_PATH", "./styles/mosaic.jpg")
        self.TRAIN_DATASET = os.environ.get("TRAIN_DATASET", "./train_dataset")
        self.GPU = False if os.environ.get("GPU", 'False') == 'False' else True


import zipfile
import wget

class CocoDataset():

    is_dataset_exist = False

    def __init__(self, root_src_dir, train_dataset_path="train", url = 'http://images.cocodataset.org/zips/train2014.zip') -> None:
        out_path = os.path.join(root_src_dir, train_dataset_path)

        if not os.path.isdir(out_path):
            print("Given directory doesn't exist")
            os.makedirs(out_path, exist_ok=True)

        if not os.listdir(out_path):
            print(f"Directory is empty, download Coco dataset from {url}")
            filename = wget.download(url, out=out_path)

            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(train_dataset_path)
            os.remove(filename)
            self.is_dataset_exist = True
        else:
            print(f"Directory is not empty, no need to donwload {url}")
            self.is_dataset_exist = True

