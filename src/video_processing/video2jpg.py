# from __future__ import print_function, division
import os
import sys
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

        # skip large files
        # if os.path.getsize(video_file_path) > maxSize * 1000:
        #	continue

        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.' + img_ext)):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.makedirs(dst_directory_path)
                else:
                    print('Folders with images have already existed')
                    continue
            else:
                os.makedirs(dst_directory_path)
        except:
            print(dst_directory_path)
            continue

        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
        ext = 'jpg'
        # -r 1 get frame every second, one frame every four seconds would be -r 0.25
        cmd = 'ffmpeg  -i \"{}\" -r 1.5 -qscale:v 2 \"{}/{}_image_{}_%05d.{}\"'.format(video_file_path,
                                                                                     dst_directory_path,
                                                                                     current_time,
                                                                                     name, img_ext)
        # cmd = 'ffmpeg -i \"{}\" -qscale:v 2 \"{}/image_{}_%05d.{}\"'.format(video_file_path, dst_directory_path,
        #                                                                          name, img_ext)

        # cmd = 'ffmpeg -i \"{}\" -r 10 -qscale:v 2 \"{}/image_{}_%05d.{}\"'.format(video_file_path, dst_directory_path,
        #                                                                     name, img_ext)

        # cmd = 'ffmpeg -i \"{}\" -qscale:v 2 \"{}/%06d.{}\"'.format(video_file_path, dst_directory_path, img_ext)

        # cmd = 'ffmpeg -i \"{}\" -qscale:v 2 \"{}/%d.{}\"'.format(video_file_path, dst_directory_path, img_ext)

        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


def train_val_split(dir_path, valid_dir_path, class_name, val_part=0.2):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    valid_class_path = os.path.join(valid_dir_path, class_name)
    if not os.path.exists(valid_class_path):
        os.makedirs(valid_class_path)

    for i, (file_name) in enumerate(os.listdir(class_path)):
        name, ext = os.path.splitext(file_name)
        train_directory_path = os.path.join(class_path, name)
        valid_directory_path = os.path.join(valid_class_path, name)

        if i % int(val_part * 100) == 0:
            shutil.move(train_directory_path, valid_directory_path)


if __name__ == "__main__":

    dir_path = os.path.join('data', 'video_processing', 'video2jpg', 'video')
    dst_dir_path = os.path.join('data', 'video_processing', 'video2jpg', 'train')
    valid_dir_path = os.path.join('data', 'video_processing', 'video2jpg', 'valid')

    types = ['.avi', '.mp4', '.webm']

    img_ext = 'jpg'

    val_part = 0

    for class_name in os.listdir(dir_path):
        class_process(dir_path, dst_dir_path, class_name, types, img_ext)
        # train_val_split(dst_dir_path, valid_dir_path, class_name, val_part)

        path = Path(dst_dir_path).joinpath(class_name, "all")  # os.path.join(dst_dir_path, class_name, "all")
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

        directories = [x[0] for x in os.walk(os.path.join(dst_dir_path, class_name))]
        Path(path).mkdir(parents=True, exist_ok=True)

        for dir in directories:
            images = get_all_files_in_folder(dir, [f"*.{img_ext}"])
            for img in tqdm(images):
                shutil.copy(img, path)
