import cv2

from pathlib import Path


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    # files_grabbed = [file[:-4] for file in files_grabbed if 'orig' in file.lower()]
    return files_grabbed


classLabels_dict = {1: "risunok", 2: "nadav", 3: "morshiny", 4: "izlom"}

image_ext = 'png'
images = get_all_files_in_folder(Path('data/manual_labeling_images/multi_label_image_classification/input_draw'),
                                 ['*.' + image_ext])

txts = get_all_files_in_folder(Path('data/manual_labeling_images/multi_label_image_classification/output'),
                               ['*.txt'])

for txt in txts:
    for im in images:
        if im.stem in txt.stem:
            images.remove(im)

images = sorted(images * len(classLabels_dict.keys()))

key_bindings = {48: 0, 49: 1, 56: 8}

image_classes = []
current_class = list(classLabels_dict.keys())[0]

for ind, image_path in enumerate(images):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img_orig = cv2.imread(str('data/manual_labeling_images/multi_label_image_classification/input/' + image_path.name), cv2.IMREAD_COLOR)

    img_draw = img.copy()
    cv2.putText(img_draw, str(classLabels_dict[current_class]), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    current_class += 1
    # img = cv2.resize(img, (768, 768))

    cv2.namedWindow(image_path.name)  # Create a named window
    cv2.moveWindow(image_path.name, 1000, 200)

    cv2.imshow(image_path.name, img_draw)

    key = cv2.waitKey()

    image_classes.append(key_bindings[key])

    if len(image_classes) == len(classLabels_dict.keys()):
        current_class = list(classLabels_dict.keys())[0]

        not_confident = ''
        founded_classes = ''
        for ii, cl in enumerate(image_classes):
            if cl == 1 or cl == 8:
                founded_classes += list(classLabels_dict.values())[ii] + ' '
            if cl == 8:
                not_confident = 'not_conf_'
        print(str(image_path.stem) + '   ------   ' + str(founded_classes))

        with open(Path('data/manual_labeling_images/multi_label_image_classification/output').joinpath(
                not_confident + image_path.stem + '.txt'), 'w') as f:
            classes_names = ''
            for key, value in classLabels_dict.items():
                classes_names += str(value) + ' '

            f.write("%s\n" % classes_names)

            # print('ddd', image_classes)
            classes_str = ' '.join([str(x) for x in image_classes])
            f.write("%s\n" % classes_str)

        image_classes = []
        cv2.imwrite(
            'data/manual_labeling_images/multi_label_image_classification/output/' + not_confident + image_path.name,
            img_orig)

    print(str(ind + 1) + ' / ' + str(len(images)))
    cv2.destroyAllWindows()
