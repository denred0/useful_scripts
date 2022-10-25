import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder, recreate_folder, plot_one_box


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y

    y = [int(a) for a in y]
    return y


def get_intersection_area(image_box, annot_box):
    x1 = max(image_box[0], annot_box[0])
    y1 = max(image_box[1], annot_box[1])
    x2 = min(image_box[2], annot_box[2])
    y2 = min(image_box[3], annot_box[3])

    intersection = 0
    if ((x2 - x1) >= 0) and ((y2 - y1) >= 0):
        intersection = (x2 - x1) * (y2 - y1)

    return intersection


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


def main(
        data_dir,
        images_ext,
        desired_size,
        output_annot_dir,
        output_vis_dir,
        filter_classes,
        classes_path,
        save_vis=True,
        min_intersection=0.4
):
    images = get_all_files_in_folder(data_dir, [f'*.{images_ext}'])
    annotations = get_all_files_in_folder(data_dir, ['*.txt'])

    assert len(images) == len(annotations), 'len(images) != len(annotations)'

    for im in tqdm(images):
        img = cv2.imread(str(im))

        h, w = img.shape[:2]

        parts_w = w // desired_size[0] + 1
        extra_w = (parts_w * desired_size[0] - w)
        one_part_inside_w = (w - extra_w) // parts_w
        one_part_extra_w = (w - one_part_inside_w * parts_w) // (parts_w - 1)

        parts_h = h // desired_size[1] + 1
        extra_h = (parts_h * desired_size[1] - h)
        one_part_inside_h = (h - extra_h) // parts_h
        one_part_extra_h = (h - one_part_inside_h * parts_h) // (parts_h - 1)

        min_x, min_y = 0, 0
        max_x, max_y = desired_size[0], desired_size[1]

        parts_images = []
        sizes = []
        for _ in (range(parts_w)):
            for _ in (range(parts_h)):
                parts_images.append([min_x, min_y, max_x, max_y])
                sizes.append((max_x - min_x) * (max_y - min_y))

                min_y = (max_y - one_part_extra_h)
                max_y = (min_y + desired_size[1])

            min_x = (max_x - one_part_extra_w)
            max_x = (min_x + desired_size[0])

            min_y = 0
            max_y = desired_size[1]

        with open(im.parent.joinpath(f'{im.stem}.txt')) as file:
            annotations = file.readlines()
            annotations = [line.rstrip() for line in annotations]

        annotations = [[float(x) for x in a.split()] for a in annotations]

        part_images_annotations = []
        for box_image in parts_images:
            image_annotations = []

            for annot_n in annotations:
                ann = xywhn2xyxy(annot_n[1:], w=w, h=h)

                intersection_area = get_intersection_area(box_image, ann)
                if intersection_area / box_area(ann) > min_intersection:
                    ann_w = ann[2] - ann[0]
                    ann_h = ann[3] - ann[1]

                    ann_x1 = ann[0] - box_image[0] if ann[0] > box_image[0] else 0
                    ann_y1 = ann[1] - box_image[1] if ann[1] > box_image[1] else 0
                    ann_x2 = ann_x1 + ann_w if ann[0] > box_image[0] else ann_x1 + ann_w - (box_image[0] - ann[0])
                    ann_y2 = ann_y1 + ann_h if ann[1] > box_image[1] else ann_y1 + ann_h - (box_image[1] - ann[1])

                    ann_x_center_norm = (ann_x1 + (ann_x2 - ann_x1) / 2) / desired_size[0]
                    ann_y_center_norm = (ann_y1 + (ann_y2 - ann_y1) / 2) / desired_size[1]
                    ann_w_norm = (ann_x2 - ann_x1) / desired_size[0]
                    ann_h_norm = (ann_y2 - ann_y1) / desired_size[1]

                    image_annotations.append(
                        [
                            int(annot_n[0]),
                            ann_x_center_norm,
                            ann_y_center_norm,
                            ann_w_norm,
                            ann_h_norm
                        ])

            part_images_annotations.append(image_annotations)

        for i, (image_box, annot_pack) in enumerate(zip(parts_images, part_images_annotations)):
            img_part = img[image_box[1]:image_box[3], image_box[0]:image_box[2]]
            cv2.imwrite(str(Path(output_annot_dir).joinpath(f'{im.stem}_{i}.{images_ext}')), img_part)

            with open(Path(output_annot_dir).joinpath(f'{im.stem}_{i}.txt'), 'w') as f:
                for an in annot_pack:
                    f.write(f"{an[0]} {an[1]} {an[2]} {an[3]} {an[4]}\n")

    if save_vis:
        plot_yolo_box(output_annot_dir, images_ext, output_vis_dir, classes_path, filter_classes)


if __name__ == '__main__':
    data_dir = 'data/cut_yolo_images/input'
    images_ext = 'jpg'

    output_annot_dir = 'data/cut_yolo_images/output/annotations'
    recreate_folder(output_annot_dir)

    output_vis_dir = 'data/cut_yolo_images/output/visualization'
    recreate_folder(output_vis_dir)
    save_vis = True
    filter_classes = ['clefG', 'clefF', 'ledgerLine', 'noteheadBlackOnLine', 'noteheadBlackInSpace']
    classes_path = 'data/cut_yolo_images/classes.txt'

    desired_size = (640, 640)  # w, h
    min_intersection = 0.5

    main(
        data_dir,
        images_ext,
        desired_size,
        output_annot_dir,
        output_vis_dir,
        filter_classes,
        classes_path,
        save_vis,
        min_intersection
    )
