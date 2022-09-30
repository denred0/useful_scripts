import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def get_all_files_in_folder(folder: str, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(Path(folder).rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def class_process(dir_path, dst_dir_path, class_name, types, img_ext, maxSize=1024):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.makedirs(dst_class_path)

    for file_name in os.listdir(class_path):
        for type in types:
            if type not in file_name:
                continue

        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path, name)

        video_file_path = os.path.join(class_path, file_name)

        if os.path.exists(dst_directory_path):
            shutil.rmtree(dst_directory_path)
        Path(dst_directory_path).mkdir(parents=True, exist_ok=True)

        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # -r 1 get frame every second, one frame every four seconds would be -r 0.25
        cmd = 'ffmpeg -i \"{}\" -r 5  -qscale:v 2 \"{}\{}_image_{}_%05d.{}\"'.format(video_file_path,
                                                                                 dst_directory_path,
                                                                                 current_time,
                                                                                 name, img_ext)

        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


if __name__ == "__main__":
    dir_path = os.path.join('data', 'video_processing', 'video2jpg', 'video')
    dst_dir_path = os.path.join('data', 'video_processing', 'video2jpg', 'images')
    valid_dir_path = os.path.join('data', 'video_processing', 'video2jpg', 'valid')

    types = ['.avi', '.mp4', '.webm', '.m4v']
    img_ext = 'jpg'

    for class_name in os.listdir(dir_path):
        class_process(dir_path, dst_dir_path, class_name, types, img_ext)

        path = Path(dst_dir_path).joinpath(class_name, "all")
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

        directories = [x[0] for x in os.walk(os.path.join(dst_dir_path, class_name))]
        Path(path).mkdir(parents=True, exist_ok=True)

        for dir in directories:
            images = get_all_files_in_folder(dir, [f"*.{img_ext}"])
            for img in tqdm(images):
                shutil.copy(img, path)
