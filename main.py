import pandas as pd
import numpy as np
from tqdm import tqdm

TRAIN_CSV_PATH = 'tensorflow-great-barrier-reef/train.csv'
TEST_CSV_PATH = 'tensorflow-great-barrier-reef/test.csv'
TRAIN_IMG_PATH = 'tensorflow-great-barrier-reef/train_images'
IMG_HEIGHT = 720
IMG_WIDTH = 1280
STARFISH_CLASS = 0


def dataset_info():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(train_df.columns)
    print(train_df.head())
    annotated_labels = (train_df.annotations.values != '[]').sum()
    print(f'Around {np.round(annotated_labels / train_df.shape[0] * 100)}% images are labeled')


def create_annotations_for_frame(annotated_frames):
    # create image path, normalized column with following format: class x_center y_center width height
    video_id = annotated_frames['video_id']
    video_frame = annotated_frames['video_frame']
    annotations = eval(annotated_frames.annotations)
    annotated_txt = open(f'{TRAIN_IMG_PATH}/video_{video_id}/{video_frame}.txt', 'w')
    for annotation in annotations:
        annotated_txt.write(f"{STARFISH_CLASS} "
                            f"{annotation['x']/IMG_WIDTH} {annotation['y']/IMG_HEIGHT} "
                            f"{annotation['width']/IMG_WIDTH} {annotation['height']/IMG_HEIGHT}\n")
    annotated_txt.close()


def prepare_dataset():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    annotated_frames = train_df[train_df.annotations.values != '[]']
    #annotated_frames.apply(lambda x: create_annotations_for_frame(x), axis=1)


prepare_dataset()
