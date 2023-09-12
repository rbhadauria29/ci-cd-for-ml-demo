import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "raw_dataset/weather.csv"
PROCESSED_DATASET = "processed_dataset/weather.csv"
RFC_FOREST_DEPTH = 2


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)
