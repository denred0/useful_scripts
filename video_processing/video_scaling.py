from __future__ import print_function, division
import os
import subprocess
from pathlib import Path


def scale_videos(dir_path, dst_dir_path, types, width, height, maxSize=1024):
    for i, (subdir, dirs, files) in enumerate(os.walk(dir_path)):
        for folder in dirs:

            Path(os.path.join(dst_dir_path, folder)).mkdir(parents=True, exist_ok=True)

            for file_name in os.listdir(os.path.join(dir_path, folder)):
                for type in types:
                    if type not in file_name:
                        continue

                name, ext = os.path.splitext(file_name)

                new_filename = os.path.join(dst_dir_path, folder) + '/' + name + '_' + str(width) + '_' + str(
                    height) + ext
                cmd = 'ffmpeg -i \"{}\" -vf scale={}:{} \"{}\"'.format(os.path.join(dir_path, folder, file_name), width,
                                                                       height, new_filename)

                print(cmd)
                subprocess.call(cmd, shell=True)
                print('\n')


if __name__ == "__main__":
    dir_path = os.path.join('data', 'video2jpg', 'scale_videos', 'to_scale')
    dst_dir_path = os.path.join('data', 'video2jpg', 'scale_videos', 'scaled')
    types = ['.avi', '.mp4']

    width = 320
    height = 240

    scale_videos(dir_path, dst_dir_path, types, width, height)
