import cv2
import imageio

from pathlib import Path

from helpers import recreate_folder, get_all_files_in_folder


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def video2frames(video_path, output_dir, images_ext):
    capture = cv2.VideoCapture(video_path)

    frameNr = 0

    while (True):
        success, frame = capture.read()
        if success:
            frame = image_resize(frame, width=200)
            cv2.imwrite(str(Path(output_dir).joinpath(f'{str(frameNr).zfill(6)}.{images_ext}')), frame)
        else:
            break
        frameNr = frameNr + 1

    capture.release()


def frames2gif(output_dir, output_dir_images, images_ext, gif_name, fps):
    filenames = get_all_files_in_folder(output_dir_images, [f'*.{images_ext}'])

    images = []
    for filename in filenames:
        images.append(imageio.v2.imread(filename))
    imageio.mimsave(Path(output_dir).joinpath(gif_name), images, fps=fps)


if __name__ == '__main__':
    video_path = 'data/video_to_gif/input/problematic_gestures.mp4'

    output_dir = 'data/video_to_gif/output'
    recreate_folder(output_dir)
    output_dir_images = 'data/video_to_gif/output/images'
    recreate_folder(output_dir_images)
    images_ext = 'jpg'

    fps = 40

    video2frames(video_path, output_dir_images, images_ext)

    gif_name = 'result.gif'
    frames2gif(output_dir, output_dir_images, images_ext, gif_name, fps)
