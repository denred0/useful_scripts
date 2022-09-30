import cv2
import os

from pathlib import Path
from tqdm import tqdm

from my_utils import get_all_files_in_folder, recreate_folder, plot_one_box, get_iou

input_dir = "data/podrydchiki/input"
output_dir = "data/podrydchiki/output"
recreate_folder(output_dir)

images = get_all_files_in_folder(input_dir, ["*.jpg"])
txts = get_all_files_in_folder(input_dir, ["*.txt"])

delta_px = 20

for im in tqdm(images):
    img = cv2.imread(str(im))

    h, w = img.shape[:2]

    with open(im.parent.joinpath(im.stem + ".txt")) as file:
        lines = file.readlines()
        lines = [line.rstrip().split() for line in lines]

    persons = []
    converted_lines = []
    for line in lines:
        label = int(line[0])

        if label == 6:  # person
            person_exist = True
            x_min_person = round(float(line[1]) * w - float(line[3]) * w / 2) - delta_px
            x_min_person = x_min_person if x_min_person > 0 else 0

            y_min_person = round(float(line[2]) * h - float(line[4]) * h / 2) - delta_px
            y_min_person = y_min_person if y_min_person > 0 else 0

            x_max_person = round(float(line[1]) * w + float(line[3]) * w / 2) + delta_px
            x_max_person = x_max_person if x_max_person < w else w

            y_max_person = round(float(line[2]) * h + float(line[4]) * h / 2) + delta_px
            y_max_person = y_max_person if y_max_person < h else h

            persons.append([x_min_person, y_min_person, x_max_person, y_max_person])

        else:
            converted_line = [label,
                              round(float(line[1]) * w - float(line[3]) * w / 2),
                              round(float(line[2]) * h - float(line[4]) * h / 2),
                              round(float(line[1]) * w + float(line[3]) * w / 2),
                              round(float(line[2]) * h + float(line[4]) * h / 2)]

            converted_lines.append(converted_line)

    if not persons:
        continue

    for i, person in enumerate(persons):
        img_crop = img[person[1]:person[3], person[0]:person[2]]
        labels = []

        for c_line in converted_lines:
            iou = get_iou(person, c_line[1:])

            person_w = person[2] - person[0]
            person_h = person[3] - person[1]

            if iou > 0:
                x_min = c_line[1] - person[0] if c_line[1] - person[0] > 0 else 0
                y_min = c_line[2] - person[1] if c_line[2] - person[1] > 0 else 0
                x_max = x_min + (c_line[3] - c_line[1]) if x_min + (
                        c_line[3] - c_line[1]) < person[2] - person[0] else person[2] - person[0]
                y_max = y_min + (c_line[4] - c_line[2]) if y_min + (
                        c_line[4] - c_line[2]) < person[3] - person[1] else person[3] - person[1]

                if (x_min == 0 or y_min == 0 or x_max == person_w or y_max == person_h) and iou < 0.1:
                    pass
                else:
                    labels.append(str(c_line[0]) + " " +
                                  str((x_min + (x_max - x_min) / 2) / person_w) + " " +
                                  str((y_min + (y_max - y_min) / 2) / person_h) + " " +
                                  str((x_max - x_min) / person_w) + " " +
                                  str((y_max - y_min) / person_h))

            if labels:
                cv2.imwrite(os.path.join(output_dir, im.stem + "_x0_" + str(person[0]) + "_y0_" + str(
                    person[1]) + "_deltapx_" + str(delta_px) + ".jpg"), img_crop)
                with open(os.path.join(output_dir, im.stem + "_x0_" + str(person[0]) + "_y0_" + str(
                        person[1]) + "_deltapx_" + str(delta_px) + ".txt"), 'w') as f:
                    for item in labels:
                        f.write("%s\n" % item)

                # img_crop = plot_one_box(img_crop, [x_min, y_min, x_max, y_max], str(c_line[0]))

        # cv2.imshow(im.stem, img_crop)
        # cv2.waitKey()
        # cv2.imwrite(os.path.join(output_dir, im.stem + str(i) + ".jpg"), img_crop)

    # converted_lines = []
    # person_exist = False
    # x_min_person, y_min_person, x_max_person, y_max_person = 0, 0, 0, 0
    # for line in lines:
    #     label = int(line[0])
    #
    #     if label == 6:  # person
    #         person_exist = True
    #         x_min_person = round(float(line[1]) * w - float(line[3]) * w / 2) - delta_px
    #         x_min_person = x_min_person if x_min_person > 0 else 0
    #
    #         y_min_person = round(float(line[2]) * h - float(line[4]) * h / 2) - delta_px
    #         y_min_person = y_min_person if y_min_person > 0 else 0
    #
    #         x_max_person = round(float(line[1]) * w + float(line[3]) * w / 2) + delta_px
    #         x_max_person = x_max_person if x_max_person < w else w
    #
    #         y_max_person = round(float(line[2]) * h + float(line[4]) * h / 2) + delta_px
    #         y_max_person = y_max_person if y_max_person < h else h
    #
    #     else:
    #         converted_line = [label,
    #                           round(float(line[1]) * w - float(line[3]) * w / 2),
    #                           round(float(line[2]) * h - float(line[4]) * h / 2),
    #                           round(float(line[1]) * w + float(line[3]) * w / 2),
    #                           round(float(line[2]) * h + float(line[4]) * h / 2)]
    #
    #         converted_lines.append(converted_line)
    #
    # if not person_exist:
    #     continue
    #
    # img_crop = img[y_min_person:y_max_person, x_min_person:x_max_person]
    # truncated_lines = []
    # for c_line in converted_lines:
    #     x_min = c_line[1] - x_min_person if c_line[1] - x_min_person > 0 else 0
    #     y_min = c_line[2] - y_min_person if c_line[2] - y_min_person > 0 else 0
    #     x_max = x_min + (c_line[3] - c_line[1]) if x_min + (
    #             c_line[3] - c_line[1]) < x_max_person - x_min_person else x_max_person - x_min_person
    #     y_max = y_min + (c_line[4] - c_line[2]) if y_min + (
    #             c_line[4] - c_line[2]) < y_max_person - y_min_person else y_max_person - y_min_person
    #
    #     # if x_min != 0 and y_min != 0 and x_max != 0 and y_max != 0:
    #     img_crop = plot_one_box(img_crop, [x_min, y_min, x_max, y_max], str(c_line[0]))
    #
    #     cv2.imshow(im.stem, img_crop)
    #     cv2.waitKey()
    # cv2.imwrite(os.path.join(output_dir, im.name), img_crop)

    # for c_line in converted_lines:
    #     img = plot_one_box(img, [c_line[1], c_line[2], c_line[3], c_line[4]], str(c_line[0]))
    # cv2.imwrite(os.path.join(output_dir, im.name), img)
