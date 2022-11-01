import shutil

from helpers import recreate_folder, get_all_files_in_folder

from pathlib import Path
from tqdm import tqdm

output = "data/copy_data/output"
recreate_folder(output)

source = get_all_files_in_folder(Path("data/copy_data/source"), ["*"])
input = get_all_files_in_folder(Path("data/copy_data/input"), ["*.jpg"])

for s in tqdm(source):
    for inp in input:
        if inp.stem == s.stem:
            shutil.copy(inp, output)
            shutil.copy(s, output)
