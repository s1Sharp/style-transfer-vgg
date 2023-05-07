import os
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
            filename = wget.download(url, out=out_path, bar=wget.bar_thermometer)

            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(train_dataset_path)
            os.remove(filename)
            self.is_dataset_exist = True
        else:
            print(f"Directory is not empty, no need to download {url}")
            self.is_dataset_exist = True
