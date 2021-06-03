import os
import sys
import subprocess
import shutil
from tqdm import tqdm
from pathlib import Path
from os import walk
import cv2


def cut_onto_img(data_dir, data_dst_img, data_dst_video, class_name, types, skip_frames, one_video_frames_count):
    current_frames_arr = []
    video_counter = 0

    _, _, filenames = next(walk(os.path.join(data_dir, class_name)))

    for i, file_name in tqdm(enumerate(sorted(filenames))):
        for type in types:
            if type not in file_name:
                continue

        skip = False
        for skip_interval in skip_frames:
            if skip_interval[0] <= i < skip_interval[1]:
                skip = True
                break

        if not skip:

            if len(current_frames_arr) == one_video_frames_count:
                folder = str(video_counter) + '_' + class_name
                Path(os.path.join(data_dst_img, class_name, folder)).mkdir(parents=True, exist_ok=True)

                for frame in current_frames_arr:
                    shutil.copy(os.path.join(data_dir, class_name, frame),
                                os.path.join(data_dst_img, class_name, folder, frame))

                current_frames_arr = []
                video_counter += 1
                current_frames_arr.append(file_name)
            else:
                current_frames_arr.append(file_name)

        else:
            current_frames_arr = []


def create_videos(data_dir, data_dst_video, video_ext, class_name, video_size=()):
    Path(os.path.join(data_dst_video, class_name)).mkdir(parents=True, exist_ok=True)

    for subdir, dirs, files in os.walk(os.path.join(data_dir, class_name)):
        for i, folder in tqdm(enumerate(dirs)):

            img_array = []
            size_shape = video_size
            for filename in sorted(os.listdir(os.path.join(subdir, folder))):
                image = cv2.imread(os.path.join(subdir, folder, filename), cv2.IMREAD_COLOR)

                if not len(video_size) == 0:
                    dim = video_size
                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                (H, W) = image.shape[:2]
                size_shape = (W, H)

                img_array.append(image)

            video_name = str(folder) + video_ext
            out = cv2.VideoWriter(os.path.join(data_dst_video, class_name, video_name), cv2.VideoWriter_fourcc(*'DIVX'),
                                  16,
                                  size_shape)

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()


if __name__ == "__main__":
    # folder name that contains images in "_source" folder
    class_name = 'case_1'

    data_dir_img = os.path.join('data', 'video_processing', 'video_cutting', '_source')
    data_dst_img = os.path.join('data', 'video_processing', 'video_cutting', 'cut_img')

    # list of intervals frames to skip. Format is [[startFrame1, endFrame1], [startFrame2, endFrame2], ...]
    skip_frames = [[0, 10], [300, 400]]

    types = ['.jpg']
    one_video_frames_count = 50  # about 2 seconds, normal fps is 24 frames/sec

    data_dir_video = os.path.join('data', 'video_processing', 'video_cutting', 'cut_img')
    data_dst_video = os.path.join('data', 'video_processing', 'video_cutting', 'cut_video')
    video_size = (330, 256)
    video_ext = '.avi'

    # cut_onto_img(data_dir=data_dir_img,
    #              data_dst_img=data_dst_img,
    #              data_dst_video=data_dst_video,
    #              class_name=class_name,
    #              types=types,
    #              skip_frames=skip_frames,
    #              one_video_frames_count=one_video_frames_count,
    #              create_videos=create_videos,
    #              video_ext=video_ext)

    create_videos(data_dir=data_dir_video,
                  data_dst_video=data_dst_video,
                  video_ext=video_ext,
                  class_name=class_name,
                  video_size=video_size)
