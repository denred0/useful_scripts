import cv2
import shutil
import pandas as pd
import numpy as np

from numpy import loadtxt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List

from helpers import get_all_files_in_folder, recreate_folder

import warnings

warnings.filterwarnings("ignore")


def train_val_split(input_dir, images_ext, split_train_dir, split_val_dir, val_part):
    # root_images_dir = input_dir.parent.absolute().joinpath('images')
    all_images = get_all_files_in_folder(str(input_dir), [f'*.{images_ext}'])
    all_txts = get_all_files_in_folder(str(input_dir), ['*.txt'])

    assert len(all_images) == len(all_txts), 'len(images) != len(annotations)'

    # print(f'Total images: {len(all_images)}')
    # print(f'Total labels: {len(all_txts)}')

    labels = []
    for txt in tqdm(all_txts, desc='Calc all txts'):
        lines = loadtxt(str(txt), delimiter=' ', unpack=False)
        if lines.shape.__len__() == 1:
            lines = [lines]

        for line in lines:
            if len(line) != 0:
                labels.append(int(line[0]))

    # classes + counts
    labels_dict = pd.DataFrame(labels, columns=["x"]).groupby('x').size().to_dict()
    all_labels = sum(labels_dict.values())
    print('labels_dict', labels_dict)

    labels_parts = []
    for key, value in labels_dict.items():
        labels_parts.append(value / all_labels)

    print('labels_parts', labels_parts)
    print('classes ', len(labels_parts))

    labels_dict[-1] = 99999999

    # assign one class to image  - rariest class
    x_all = []
    labels_all = []
    for txt in tqdm(all_txts, desc='Finding the best category'):
        lines = loadtxt(str(txt), delimiter=' ', unpack=False)
        if lines.shape.__len__() == 1:
            lines = [lines]

        lab = [int(line[0]) for line in lines if len(line) != 0]

        best_cat = -1
        x_all.append(txt.stem)
        for l in lab:
            if labels_dict[l] < labels_dict[best_cat]:
                best_cat = l
        labels_all.append(best_cat)

    # stratify
    X_train, X_test, y_train, y_test = train_test_split(
        x_all,
        labels_all,
        test_size=val_part,
        random_state=42,
        shuffle=True
    )

    # copy images and txts
    for name in tqdm(X_train, desc="Copying train images"):
        shutil.copy(input_dir.joinpath(name + f'.{images_ext}'), split_train_dir)
        shutil.copy(input_dir.joinpath(name + '.txt'), split_train_dir)

    for name in tqdm(X_test, desc='Copying val images'):
        shutil.copy(input_dir.joinpath(name + f'.{images_ext}'), split_val_dir)
        shutil.copy(input_dir.joinpath(name + '.txt'), split_val_dir)

    # check stratification
    all_txt_train = get_all_files_in_folder(split_train_dir, ['*.txt'])

    # collect train classes and compare with all classes
    labels_train = []
    for txt in tqdm(all_txt_train, desc="Calc statistics"):
        lines = loadtxt(str(txt), delimiter=' ', unpack=False)
        if lines.shape.__len__() == 1:
            lines = [lines]

        for line in lines:
            if len(line) != 0:
                labels_train.append(line[0])

    labels_train_dict = pd.DataFrame(labels_train, columns=["x"]).groupby('x').size().to_dict()

    st = []
    labels_dict.pop(-1)
    for key, value in labels_dict.items():
        val = labels_train_dict[key] / value
        st.append(val)

        print(f'Class {key} | counts {value} | val_part {val}')

    print('Train part:', np.mean(st))


if __name__ == '__main__':
    input_dir = Path('data/train_val_split/dataset')
    images_ext = 'jpg'

    split_train_dir = 'data/train_val_split/output/train'
    recreate_folder(split_train_dir)

    split_val_dir = 'data/train_val_split/output/val'
    recreate_folder(split_val_dir)

    val_part = 0.25

    train_val_split(input_dir, images_ext, split_train_dir, split_val_dir, val_part)
