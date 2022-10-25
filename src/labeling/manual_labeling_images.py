import os
import shutil
import time
from glob import glob

from pathlib import Path

import cv2


def get_move_dir_name(root, class_idx, class_name):
    return f'{root}/{str(class_idx).zfill(2)}_{class_name}'


def prepare_folders(root, classes_template):
    for k, v in classes_template.items():
        dir_name = get_move_dir_name(root, k, v)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def get_settings(from_user=False):
    if from_user:
        print('Давайте настроим текущий запуск. Файл должен лежать в корневой папке, откуда нужно все рассортировать')
        num_defects = int(input(f'Введите сколько есть всего дефектов: '))
        classes_template = {0: 'nodefect'}
        for i in range(1, num_defects + 1):
            classes_template[i] = input(f'Введите название {i} дефекта: ').lower()
        print('Вы установили следующие виды классов:')
        class_message = ''
        for k, v in classes_template.items():
            class_message += f'{k} - {v}\n'

        print(class_message)
        return num_defects, classes_template, class_message
    else:
        # classes_template = {0: 'nodefect', 1: 'plasticade', 2: 'dirty', 3: 'otherdefect', 4: 'uncertain'}
        classes_template = {0: 'good', 1: 'bad', 2: 'other'}

        num_defects = len(classes_template)
        print('У вас установлены следующие виды классов:')
        class_message = ''
        for k, v in classes_template.items():
            class_message += f'{k} - {v}\n'

        print(class_message)
        return num_defects, classes_template, class_message


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    # files_grabbed = [file[:-4] for file in files_grabbed if 'orig' in file.lower()]
    return files_grabbed


def delete_object_duplicates(types, orig_suffix, changed_suffix):
    files_grabbed = get_all_files_in_folder(types)
    last_obj = None
    deleted_objs = 0
    for file in files_grabbed:
        current_obj = file.split('_')[0]
        if current_obj == last_obj:
            os.remove(file + orig_suffix)
            print(f'Deleted object duplicate: {file + orig_suffix}')
            os.remove(file + changed_suffix)
            print(f'Deleted object duplicate: {file + changed_suffix}')
            deleted_objs += 1
        else:
            last_obj = current_obj
    print(f'Deleted {deleted_objs} object duplicates')


def delete_file_duplicates(folder, changed_suffix, orig_suffix):
    all_files_with_crops = sorted(list(folder.rglob('*' + changed_suffix)))
    #   all_files_with_crops = folder.rglob('*' + changed_suffix)
    removed = 0
    for file in all_files_with_crops:
        os.remove(file)
        removed += 1

    print(f'Deleted {removed} file duplicates')


def main():
    key_bindings = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9}

    types = ('*.jpeg', '*.jpg', '*.png')
    orig_suffix = '_Changed_Orig.png'
    changed_suffix = '_Changed.jpg'
    root = '.'
    camera = 'fire'
    window_width = 400

    folder_sorted = Path('data').joinpath('manual_labeling_images').joinpath('sorted_folder').joinpath(camera)
    folder_to_sort = Path('data').joinpath('manual_labeling_images').joinpath('to_sort').joinpath(camera)

    # delete_object_duplicates(types, orig_suffix, changed_suffix)
    delete_file_duplicates(folder_to_sort, changed_suffix, orig_suffix)
    files_grabbed = get_all_files_in_folder(folder_to_sort, types)
    num_files = len(files_grabbed)

    num_defects, classes_template, class_message = get_settings()

    counter = 1
    times = []
    start = time.time()

    prepare_folders(folder_sorted, classes_template)

    for file in files_grabbed:

        orig_file_path = f'{root}/{file}'

        orig_img = cv2.imread(orig_file_path, cv2.IMREAD_COLOR)
        orig_img = resize_with_aspect_ratio(orig_img, width=window_width)

        # white rectangle
        # cv2.rectangle(orig_img, (0, 0), (235, 375), (255, 255, 255), -1)

        x_start = 10
        y_start = 30
        y_interval = 30

        font_size = 0.7
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)

        for i, (key, value) in enumerate(classes_template.items()):
            cv2.putText(orig_img, str(key) + ' - ' + value,
                        (x_start, y_start + y_interval * i), font, font_size, color, thickness)

        files_grabbed_list = []
        for key, value in classes_template.items():
            images = get_all_files_in_folder(folder_sorted.joinpath(str(key).zfill(2) + '_' + value), types)
            files_grabbed_list.append(len(images))

        # summary
        y_start_summary = 210
        color_summary = (0, 255, 0)
        total_files = len(get_all_files_in_folder(folder_to_sort, types))
        cv2.putText(orig_img, 'TOTAL: ' + str(total_files), (x_start, y_start_summary), font,
                    font_size, color_summary, thickness)

        for i, (key, value) in enumerate(classes_template.items()):
            cv2.putText(orig_img, value + '  ' + str(files_grabbed_list[i]),
                        (x_start, y_start_summary + y_interval * (i + 1)),
                        font,
                        font_size, color_summary, thickness)

        cv2.imshow(orig_file_path, orig_img)
        print(class_message)
        print(f'{counter}/{num_files}, {round(counter/num_files * 100, 3)}%')

        key = cv2.waitKey()
        class_input = key_bindings[key]

        dest_dir = get_move_dir_name(folder_sorted, class_input, classes_template[class_input])
        shutil.move(orig_file_path, dest_dir)
        print(f'Успешно перемещены:\n{dest_dir}/{file}')
        counter += 1
        times.append(time.time() - start)
        start = time.time()

        mean_op_time = sum(times) / len(times)
        print(
            f'Среднее время операции: {mean_op_time:0.2f}. Осталось {((num_files - counter) * mean_op_time) // 60} min.')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    print("Все файлы успешно обработаны!")
