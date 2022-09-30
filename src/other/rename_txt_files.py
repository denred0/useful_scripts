import shutil
import os

from tqdm import tqdm
from my_utils import recreate_folder, get_all_files_in_folder

input_dir = "data/rename_txt_files/input"
output_dir = "data/rename_txt_files/output"
recreate_folder(output_dir)

txts = get_all_files_in_folder(input_dir, ["*jpg"])

for txt in tqdm(txts):
    txt_name = str(int(txt.stem) - 1).zfill(6) + ".jpg"
    shutil.copy(txt, os.path.join(output_dir, txt_name))
