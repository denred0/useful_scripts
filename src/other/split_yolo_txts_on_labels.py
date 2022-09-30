import shutil
import os

from pathlib import Path
from tqdm import tqdm

from my_utils import get_all_files_in_folder, recreate_folder


def split_yolo_txts_on_labels(input_dir: str, output_dir: str, ext_image: str) -> None:
    txt_paths = get_all_files_in_folder(input_dir, ["*.txt"])

    for txt_path in tqdm(txt_paths):

        with open(txt_path) as file:
            lines = [line.rstrip() for line in file.readlines()]

        labels_dict = {}
        for line in lines:
            label = line.split()[0]

            labels_dict[label] = labels_dict[label] + [line] if label in labels_dict else labels_dict.setdefault(label,
                                                                                                                 [line])

        for (key, value) in labels_dict.items():
            Path(output_dir).joinpath(key).mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(os.sep.join(str(txt_path).split(os.sep)[:-1])).joinpath(txt_path.stem + "." + ext_image),
                        Path(output_dir).joinpath(key))

            with open(Path(output_dir).joinpath(key).joinpath(txt_path.name), 'w') as f:
                for line in value:
                    f.write("%s\n" % str(line))


if __name__ == "__main__":
    input_dir = "data/split_yolo_txts_on_labels/input"
    ext_image = "jpg"

    output_dir = "data/split_yolo_txts_on_labels/output"
    recreate_folder(output_dir)

    split_yolo_txts_on_labels(input_dir, output_dir, ext_image)
