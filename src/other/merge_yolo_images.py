import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from helpers import get_all_files_in_folder, recreate_folder, xywhn2xyxy, xyxy2xywhn_single


def get_image_size(cut_size_x, cut_size_y, max_ind_x, max_ind_y, extra_part_x, extra_part_y, h, w):
    min_x = 0
    min_y = 0
    max_x = w if max_ind_x == 0 else cut_size_x
    max_y = h if max_ind_y == 0 else cut_size_y
    for i in range(max_ind_x):
        for j in range(max_ind_y):
            min_y = max_y - extra_part_y
            add_y = h if max_ind_y == 0 else cut_size_y
            max_y = min_y + add_y

        min_x = max_x - extra_part_x
        add_x = w if max_ind_x == 0 else cut_size_x
        max_x = min_x + add_x

        min_y = 0
        if i != max_ind_x - 1:
            max_y = h if max_ind_y == 0 else cut_size_y

    return max_x, max_y


def get_transformed_annotation(an, cut_size_x, cut_size_y, min_x, min_y, w, h):
    an = an.split()
    an = [float(a) for a in an]
    an_xyxy = xywhn2xyxy(an[1:], w=cut_size_x, h=cut_size_y)

    an_xyxy_w = an_xyxy[2] - an_xyxy[0]
    an_xyxy_h = an_xyxy[3] - an_xyxy[1]

    x1_new = an_xyxy[0] + min_x
    y1_new = an_xyxy[1] + min_y
    x2_new = x1_new + an_xyxy_w
    y2_new = y1_new + an_xyxy_h

    an_new_xyxyn = xyxy2xywhn_single(
        np.asarray([x1_new, y1_new, x2_new, y2_new], dtype=np.float64),
        w=w,
        h=h
    )

    return [int(an[0]), an_new_xyxyn[0], an_new_xyxyn[1], an_new_xyxyn[2], an_new_xyxyn[3]]


def main(input_dir, images_ext, output_dir):
    images = get_all_files_in_folder(input_dir, [f'*.{images_ext}'])
    annotations = get_all_files_in_folder(input_dir, ['*.txt'])

    assert len(images) == len(annotations), 'len(images) != len(annotations)'

    unique_images = list(set([a.stem.split('_')[:-6][0] for a in images]))

    for unique_im in tqdm(unique_images):

        parts = {}
        max_ind_x = 0
        max_ind_y = 0
        cut_size_x = 0
        cut_size_y = 0
        extra_part_x = 0
        extra_part_y = 0

        image_annotations = []

        for im in images:
            img_name = im.stem.split('_')[:-6][0]

            if img_name == unique_im:
                x_ind = int(im.stem.split('_')[-2])
                max_ind_x = max(max_ind_x, x_ind)
                y_ind = int(im.stem.split('_')[-1])
                max_ind_y = max(max_ind_y, y_ind)

                cut_size_x = int(im.stem.split('_')[-6])
                cut_size_y = int(im.stem.split('_')[-5])
                extra_part_x = int(im.stem.split('_')[-4])
                extra_part_y = int(im.stem.split('_')[-3])

                parts[f'{str(x_ind)}_{str(y_ind)}'] = im

        min_x = 0
        min_y = 0

        one_part = cv2.imread(str(parts['0_0']))
        max_x = one_part.shape[1] if max_ind_x == 0 else cut_size_x
        max_y = one_part.shape[0] if max_ind_y == 0 else cut_size_y

        w, h = get_image_size(
            cut_size_x,
            cut_size_y,
            max_ind_x,
            max_ind_y,
            extra_part_x,
            extra_part_y,
            one_part.shape[0],
            one_part.shape[1]
        )
        blank_image = np.zeros((h, w, 3), np.uint8)
        for i in range(max_ind_x + 1):
            for j in range(max_ind_y + 1):
                path_img_part = parts[f'{str(i)}_{str(j)}']

                img = cv2.imread(str(path_img_part))
                blank_image[min_y:max_y, min_x:max_x] = img

                with open(path_img_part.parent.joinpath(f'{path_img_part.stem}.txt')) as file:
                    annotations = file.readlines()
                    annotations = [line.rstrip() for line in annotations]

                for an in annotations:
                    transformed_annotation = get_transformed_annotation(
                        an,
                        one_part.shape[1] if max_ind_x == 0 else cut_size_x,
                        one_part.shape[0] if max_ind_y == 0 else cut_size_y,
                        min_x,
                        min_y,
                        w,
                        h
                    )

                    image_annotations.append(transformed_annotation)

                min_y = max_y - extra_part_y
                max_y = min_y + cut_size_y

            min_x = max_x - extra_part_x
            max_x = min_x + cut_size_x

            min_y = 0
            max_y = cut_size_y

        cv2.imwrite(str(Path(output_dir).joinpath(f'{unique_im}.{images_ext}')), blank_image)

        with open(Path(output_dir).joinpath(f'{unique_im}.txt'), 'w') as f:
            for annot in image_annotations:
                f.write(f"{annot[0]} {annot[1]} {annot[2]} {annot[3]} {annot[4]}\n")


if __name__ == '__main__':
    input_dir = 'data/merge_yolo_images/input'
    images_ext = 'jpg'

    output_dir = 'data/merge_yolo_images/output'
    recreate_folder(output_dir)

    main(input_dir, images_ext, output_dir)
