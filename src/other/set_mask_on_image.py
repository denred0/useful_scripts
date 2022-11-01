import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from helpers import recreate_folder, get_all_files_in_folder


def set_mask_on_image(template_mask, input_dir: str, output_dir: str, ext_image: str):
    txts_paths = get_all_files_in_folder(input_dir, ["*.txt"])

    for txt_path in tqdm(txts_paths):

        image = cv2.imread(str(Path(input_dir).joinpath(txt_path.stem + "." + ext_image)), cv2.IMREAD_GRAYSCALE)
        image_rgb = cv2.imread(str(Path(input_dir).joinpath(txt_path.stem + "." + ext_image)), cv2.IMREAD_COLOR)

        assert image.shape == template_mask.shape, "Shapes of image and template should be equal"

        h, w = image.shape[:2]

        with open(txt_path) as file:
            lines = [line.rstrip() for line in file.readlines()]

        lines_itog = []
        for line in lines:
            image_mask = np.zeros((h, w), dtype=np.uint8)
            data = [float(x) for x in line.split()]
            # label, x_center_norm, y_center_norm, width_norm, height_norm = line.split()

            x1 = np.clip(int(data[1] * w - (data[3] * w) / 2), 0, w)
            y1 = np.clip(int(data[2] * h - (data[4] * h) / 2), 0, h)
            x2 = np.clip(int(data[1] * w + (data[3] * w) / 2), 0, w)
            y2 = np.clip(int(data[2] * h + (data[4] * h) / 2), 0, h)

            image_mask[y1:y2, x1:x2] = 1

            intersecton = image_mask * template_mask

            if np.sum(intersecton) > np.sum(image_mask) // 2:
                y_min1 = np.min(np.nonzero(intersecton[:, x1]))
                y_max1 = np.max(np.nonzero(intersecton[:, x1]))

                y_min2 = np.min(np.nonzero(intersecton[:, (x2 - 1)]))
                y_max2 = np.max(np.nonzero(intersecton[:, (x2 - 1)]))

                y1_calc = max(y_min1, y_min2)
                y2_calc = min(y_max1, y_max2)

                w_norm = (x2 - x1) / w
                h_norm = (y2_calc - y1_calc) / h
                x_center_norm = x1 / w + w_norm / 2
                y_center_norm = y1_calc / h + h_norm / 2

                line = str(int(data[0])) + " " + str(x_center_norm) + " " + str(y_center_norm) + " " + str(
                    w_norm) + " " + str(h_norm)

                lines_itog.append(line)

        if lines_itog:
            with open(Path(output_dir).joinpath(txt_path.name), 'w') as f:
                for line in lines_itog:
                    f.write("%s\n" % str(line))

            masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=template_mask)

            cv2.imwrite(str(Path(output_dir).joinpath(txt_path.stem + "." + ext_image)), masked_image)


if __name__ == "__main__":
    input_dir = "data/set_mask_on_image/input"
    ext_image = "jpg"

    output_dir = "data/set_mask_on_image/output"
    recreate_folder(output_dir)

    template_image = cv2.imread("data/set_mask_on_image/template/001.jpg", cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(template_image, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=thresh, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=100)

    template_mask = np.clip(thresh, 0, 1)
    # cv2.imshow('Binary image', cv2.resize(thresh, (1024, 768)))
    # cv2.waitKey(0)

    set_mask_on_image(template_mask, input_dir, output_dir, ext_image)
