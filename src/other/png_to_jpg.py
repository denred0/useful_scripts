import cv2
import shutil

from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder, recreate_folder


def main(input_dir, output_dir, copy_txt=True):
    images = get_all_files_in_folder(input_dir, ['*'])

    for im in tqdm(images):
        ext = im.name.split('.')[-1]

        if ext == 'png':
            img = cv2.imread(str(im))
            cv2.imwrite(str(Path(output_dir).joinpath(f'{im.stem}.jpg')), img)
        elif ext == 'jpg':
            shutil.copy(im, output_dir)

        if copy_txt:
            shutil.copy(im.parent.joinpath(f'{im.stem}.txt'), output_dir)


if __name__ == '__main__':
    input_dir = 'data/png_to_jpg/input'
    output_dir = 'data/png_to_jpg/output'
    recreate_folder(output_dir)

    main(input_dir, output_dir)
