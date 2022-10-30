import os
import shutil

from pathlib import Path
from tqdm import tqdm
from pathlib import Path

from my_utils import get_all_files_in_folder, recreate_folder


def merge_txts_labels(input_dir: str, output_dir: str, image_ext: str) -> None:
    # label_folders = sorted([int(x) for x in os.listdir(input_dir)])
    label_folders = os.listdir(input_dir)

    all_txts_paths = get_all_files_in_folder(input_dir, ["*.txt"])
    unique_txt_files = list(set([file.name for file in all_txts_paths]))

    for file in tqdm(unique_txt_files):
        result = []
        images = []

        for label_folder in label_folders:
            file_path = Path(input_dir).joinpath(label_folder).joinpath(file)
            if os.path.isfile(file_path):
                with open(file_path) as txt_file:
                    lines = [line.rstrip() for line in txt_file.readlines()]

                result.extend(lines)

                images.append(Path(input_dir).joinpath(label_folder).joinpath(file.split(".")[0] + "." + image_ext))

        with open(Path(output_dir).joinpath(file), 'w') as f:
            for line in result:
                f.write("%s\n" % str(line))

        shutil.copy(images[0], Path(output_dir))


if __name__ == "__main__":
    input_dir = "data/merge_txts_labels/input"
    ext_image = "png"

    output_dir = "data/merge_txts_labels/output"
    recreate_folder(output_dir)

    merge_txts_labels(input_dir, output_dir, ext_image)
