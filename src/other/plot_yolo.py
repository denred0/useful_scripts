import cv2
from pathlib import Path
from tqdm import tqdm

from helpers import plot_one_box, get_all_files_in_folder, recreate_folder


def plot_yolo_box(input_images_dir, images_ext, output_dir, classes_path, filter_classes):
    images = get_all_files_in_folder((input_images_dir), [f'*.{images_ext}'])

    for im in tqdm(images):

        img = cv2.imread(str(im))
        h, w = img.shape[:2]

        with open(classes_path) as file:
            classes = file.readlines()
            classes = [line.rstrip() for line in classes]

        with open(Path(input_images_dir).joinpath(f'{im.stem}.txt')) as file:
            bboxes = file.readlines()
            bboxes = [line.rstrip() for line in bboxes]

        for box_str in bboxes:
            box = [float(x) for x in box_str.split()]

            label_num = str(int(box[0]))
            label = classes[int(box[0]) - 1]

            if filter_classes is not None and label not in filter_classes:
                continue

            xmin = int((box[1] - box[3] / 2) * w)
            ymin = int((box[2] - box[4] / 2) * h)
            xmax = int((box[1] + box[3] / 2) * w)
            ymax = int((box[2] + box[4] / 2) * h)

            img = plot_one_box(img, [xmin, ymin, xmax, ymax], label_num, [255, 0, 0], 1)

        cv2.imwrite(str(Path(output_dir).joinpath(im.name)), img)


if __name__ == '__main__':
    input_images_dir = 'data/plot_yolo/input/annotations'
    images_ext = 'jpg'

    output_dir = 'data/plot_yolo/output'
    recreate_folder(output_dir)

    classes_path = 'data/plot_yolo/input/classes.txt'

    filter_classes = None

    # filter_classes = ['clefG', 'clefF', 'ledgerLine', 'noteheadBlackOnLine', 'noteheadBlackInSpace', 'line']

    plot_yolo_box(input_images_dir, images_ext, output_dir, classes_path, filter_classes)
