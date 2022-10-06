import os
import shutil
import argparse
from collections import defaultdict

from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder, recreate_folder, read_config, intersection_over_union_box


def supplement_images_txt(project_name, ext):
    root_dir = Path('data').joinpath('verify_markup').joinpath(project_name)
    dataset_dir = root_dir.joinpath('dataset')
    recreate_folder(dataset_dir)

    all_files = []
    classes_dir = [name for name in os.listdir(root_dir.joinpath('labels'))]
    supplemented_dir = root_dir.joinpath('labels_supplemented')
    for label in tqdm(classes_dir, desc="Recreating folders"):
        recreate_folder(supplemented_dir.joinpath(label))

    for label in tqdm(classes_dir, desc="Collecting dataset"):
        images = get_all_files_in_folder(str(root_dir.joinpath('labels').joinpath(label)), [f"*.{ext}"])
        txts = get_all_files_in_folder(str(root_dir.joinpath('labels').joinpath(label)), [f"*.txt"])
        for im, txt in zip(images, txts):
            shutil.copy(im, dataset_dir)
            shutil.copy(txt, dataset_dir)

        all_files.extend([im.stem for im in images])

    all_files = list(set(all_files))

    for label in tqdm(classes_dir, desc="Supplement images and txts"):
        images = get_all_files_in_folder(str(root_dir.joinpath('labels').joinpath(label)), [f"*.{ext}"])
        txts = get_all_files_in_folder(str(root_dir.joinpath('labels').joinpath(label)), [f"*.txt"])

        for im, txt in zip(images, txts):
            shutil.copy(im, supplemented_dir.joinpath(label))
            shutil.copy(txt, supplemented_dir.joinpath(label))

        images = [im.stem for im in images]
        missing_files = list(set(all_files) - set(images))
        for f in missing_files:
            shutil.copy(dataset_dir.joinpath(f"{f}.{ext}"), supplemented_dir.joinpath(label))
            open(supplemented_dir.joinpath(label).joinpath(f"{f}.txt"), 'a').close()


def merge_txts_labels(project_name: str, ext_images) -> None:
    root_dir = os.path.join('data', "verify_markup", project_name)
    merge_config = read_config(os.path.join(root_dir, "merge_config.yaml"))
    classes_to_merge = merge_config['classes_to_merge']

    # labeling_config = read_config(os.path.join("data", project_name, "labeling_config.yaml"))

    classes_file = 'classes.txt'
    classes_file_path = os.path.join(root_dir, classes_file)
    with open(classes_file_path) as file:
        classes = {k: v for (k, v) in enumerate([line.rstrip() for line in file])}

    if not classes_to_merge:
        classes_to_merge = list(classes.values())

    dataset_dir = os.path.join(root_dir, "merge", "dataset")
    recreate_folder(dataset_dir)

    iou = merge_config['iou']
    iou_folder = "1_high_iou"
    iou_dir = os.path.join(root_dir, "merge", iou_folder)
    recreate_folder(iou_dir)

    obligatory_classes = merge_config['obligatory_classes']
    without_obligatory_classes_folder = "2_without_obligatory_classes"
    obligatory_classes_dir = os.path.join(root_dir, "merge", without_obligatory_classes_folder)
    recreate_folder(obligatory_classes_dir)

    empty_images_folder = "3_empty_images"
    empty_images_dir = os.path.join(root_dir, "merge", empty_images_folder)
    recreate_folder(empty_images_dir)

    all_txts = []
    count_of_classes = {}
    count_of_items = 0
    for ind, cl in tqdm(classes.items(), desc="Reading classes"):
        if cl in classes_to_merge:
            txts = get_all_files_in_folder(
                os.path.join(root_dir, "labels_supplemented", str(ind) + "_" + cl), ["*.txt"])
            imgs = get_all_files_in_folder(
                os.path.join(root_dir, "labels_supplemented", str(ind) + "_" + cl), [f"*.{ext_images}"])

            if len(txts) != len(imgs):
                print(f"Count of images and txts for class <<{cl}>> is not equal")
                txts_names = [txt.stem for txt in txts]
                imgs_names = [img.stem for img in imgs]
                if len(txts_names) > len(imgs_names):
                    print("Bad txts:")
                    bad_txts = list(set(txts_names) - set(imgs_names))
                    print(bad_txts)
                else:
                    print("Bad imgs:")
                    bad_imgs = list(set(imgs_names) - set(txts_names))
                    print(bad_imgs)

                return

            if count_of_items == 0:
                count_of_items = len(txts) + len(imgs)
            else:
                if count_of_items != len(txts) + len(imgs):
                    print(f"Count of items in class <<{cl}>> is differerent from others")
                    return

            all_txts.extend(txts)

    # collect classes markup by images and count dublicates
    dublicates = 0
    result = defaultdict(list)
    for txt in tqdm(all_txts, desc="Reading txts"):
        result.setdefault(txt.stem, [])

        with open(txt) as txt_file:
            lines = [line.rstrip() for line in txt_file.readlines()]

        non_empty_lines = []
        for line in lines:
            if line != "":
                non_empty_lines.append(line)

        if not non_empty_lines:
            continue

        if len(set(lines)) != len(lines):
            dublicates += 1
            with open(txt, 'w') as f:
                for line in set(lines):
                    f.write("%s\n" % str(line))

        label = str(txt).split(os.sep)[-2].split("_")[0]

        for line in set(lines):
            row = line.split(" ")
            row[0] = label
            result[txt.stem].append(" ".join(row))

    without_obligatory_classes_count = 0
    high_iou_count = 0
    empty_images_count = 0

    # saving merging results
    for filename, labels in tqdm(result.items(), desc="Saving"):
        mandatory_class_exist = False
        high_iou_exist = False
        labels = sorted(list(set(labels)))
        with open(os.path.join(root_dir, "merge", "dataset", f"{filename}.txt"), 'w') as f:
            for i in range(len(labels)):
                f.write("%s\n" % str(labels[i]))

                # calc statistics
                count_of_classes[int(labels[i].split()[0])] = count_of_classes.get(int(labels[i].split()[0]), 0) + 1

                # check obligatory classes
                for obligatory_classes_list in obligatory_classes:
                    if classes[int(labels[i].split(" ")[0])] in obligatory_classes_list:
                        mandatory_class_exist = True

                # check classes with high iou
                for j in range(i + 1, len(labels)):
                    if labels[i] and labels[j]:
                        iou_p = intersection_over_union_box([float(x) for x in labels[i].split(" ")[1:]],
                                                            [float(x) for x in labels[j].split(" ")[1:]])
                        if iou_p >= iou:
                            high_iou_exist = True

        shutil.copy(os.path.join(root_dir, "dataset", f"{filename}.{ext_images}"), dataset_dir)

        if not labels:
            empty_images_count += 1
            shutil.copy(os.path.join(root_dir, "merge", "dataset", f"{filename}.txt"),
                        os.path.join(root_dir, "merge", empty_images_folder))
            shutil.copy(os.path.join(root_dir, "dataset", f"{filename}.{ext_images}"),
                        os.path.join(root_dir, "merge", empty_images_folder))

        if high_iou_exist:
            high_iou_count += 1
            shutil.copy(os.path.join(root_dir, "merge", "dataset", f"{filename}.txt"),
                        os.path.join(root_dir, "merge", iou_folder))
            shutil.copy(os.path.join(root_dir, "dataset", f"{filename}.{ext_images}"),
                        os.path.join(root_dir, "merge", iou_folder))

        if not mandatory_class_exist and obligatory_classes:
            without_obligatory_classes_count += 1
            shutil.copy(os.path.join(root_dir, "merge", "dataset", f"{filename}.txt"),
                        os.path.join(root_dir, "merge", without_obligatory_classes_folder))
            shutil.copy(os.path.join(root_dir, "dataset", f"{filename}.{ext_images}"),
                        os.path.join(root_dir, "merge", without_obligatory_classes_folder))

    shutil.copy(os.path.join(root_dir, classes_file), os.path.join(root_dir, "merge"))

    print(f"\nFixed dublicates: {dublicates} ")
    print(f"High IoU count: {high_iou_count}")
    print(f"Without obligatory classes count: {without_obligatory_classes_count}")
    print(f"Empty images count: {empty_images_count}")

    print("\nCount of labels:")
    for cl, count in {k: v for k, v in
                      sorted(count_of_classes.items(), key=lambda item: item[1], reverse=True)}.items():
        print(f"{classes[cl]}: {count}")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name', type=str, help='project folder name')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # opt = parse_opt()
    # project_name = opt.project_name

    project_name = 'furniture'
    ext = 'jpg'

    # supplement_images_txt(project_name, ext)
    merge_txts_labels(project_name, ext)
