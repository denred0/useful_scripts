import os
import argparse
from collections import defaultdict

from tqdm import tqdm

from helpers import get_all_files_in_folder, read_config
from verify_markup import merge_txts_labels


def update_and_merge(project_name: str):
    root_dir = os.path.join('data', "verify_markup", project_name)

    upd_types = ['iou', 'obl', 'emp']

    classes_file = 'classes.txt'
    classes_file_path = os.path.join(root_dir, classes_file)
    with open(classes_file_path) as file:
        classes = {k: v for (k, v) in enumerate([line.rstrip() for line in file])}

    for upd_type in upd_types:
        if upd_type == "iou":
            dir = os.path.join(root_dir, "merge", "1_high_iou")
        elif upd_type == "obl":
            dir = os.path.join(root_dir, "merge", "2_without_obligatory_classes")
        elif upd_type == "emp":
            dir = os.path.join(root_dir, "merge", "3_empty_images")

        txts = get_all_files_in_folder(dir, ["*.txt"])

        for txt in tqdm(txts, desc="Updating txts"):
            with open(txt) as txt_file:
                lines = [line.rstrip() for line in txt_file.readlines()]

            # recreate txt for every class
            for cl in classes:
                open(os.path.join(root_dir, "labels_supplemented", str(cl) + "_" + classes[cl], txt.name), 'w')

            txt_dict = defaultdict(list)
            for line in lines:
                txt_dict[int(line.split()[0])].append(line)

            # fill txt with new values
            for cl, val in txt_dict.items():
                with open(os.path.join(root_dir, "labels_supplemented", str(cl) + "_" + classes[cl], txt.name),
                          'w') as f:
                    for line in val:
                        row = line.split(" ")
                        row[0] = '0'
                        f.write("%s\n" % str(" ".join(row)))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    parser.add_argument('upd', type=str, help='Type of update')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # opt = parse_opt()
    # project_name = opt.project_name
    # upd_type = opt.upd

    project_name = 'furniture'
    ext = 'jpg'

    update_and_merge(project_name)
    merge_txts_labels(project_name, ext)
