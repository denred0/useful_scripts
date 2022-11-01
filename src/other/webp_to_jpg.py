import cv2
import shutil

from tqdm import tqdm
from pathlib import Path

from helpers import recreate_folder, get_all_files_in_folder


def main(input_dir, output_dir):
    files = get_all_files_in_folder(input_dir, ['*'])

    for f in tqdm(files):
        ext = f.name.split('.')[-1]

        if ext == 'jpg':
            shutil.copy(f, output_dir)
        elif ext == 'webp':
            img = cv2.imread(str(f))

            retval, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])

            img = cv2.imdecode(buf, 1)
            cv2.imwrite(str(Path(output_dir).joinpath(f'{f.stem}.jpg')), img)


if __name__ == '__main__':
    input_dir = 'data/webp_to_jpg/input'

    output_dir = 'data/webp_to_jpg/output'
    recreate_folder(output_dir)

    main(input_dir, output_dir)
