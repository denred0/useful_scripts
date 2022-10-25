import shutil
import os

from pathlib import Path
from tqdm import tqdm

from my_utils import get_all_files_in_folder, recreate_folder


def rewrite_label(input_dir: str, output_dir: str, new_label: str, ext_image: str) -> None:
    txt_paths = get_all_files_in_folder(input_dir, ["*.txt"])

    for txt_path in tqdm(txt_paths):

        with open(txt_path) as file:
            lines = [line.rstrip() for line in file.readlines()]

        new_lines = []
        for line in lines:
            label = line.split()[0]

            newline = new_label + " " + " ".join(line.split()[1:])
            new_lines.append(newline)


        with open(Path(output_dir).joinpath(txt_path.name), 'w') as f:
            for line in new_lines:
                f.write("%s\n" % str(line))

        shutil.copy(txt_path.parent.joinpath(f"{txt_path.stem}.{ext_image}"), output_dir)


if __name__ == "__main__":
    input_dir = "data/rewrite_label/input"
    ext_image = "jpg"

    output_dir = "data/rewrite_label/output"
    recreate_folder(output_dir)

    new_label = "7"
    rewrite_label(input_dir, output_dir, new_label, ext_image)
