import shutil

from pathlib import Path
from tqdm import tqdm

from helpers import recreate_folder, get_all_files_in_folder

output_folder = "data/rename_files/output"
recreate_folder(output_folder)

files = get_all_files_in_folder("data/rename_files/input", ["*"])

for file in tqdm(files):
    shutil.copy(file, output_folder + "/" + "v_" + file.name)
