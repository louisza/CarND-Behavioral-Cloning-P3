from moviepy.editor import ImageSequenceClip
import argparse
import cv2

def process_img_for_visualization(image, angle, frame):
    '''
    Used by visualize_dataset method to format image prior to displaying. Converts colorspace back to original BGR, applies text to display steering angle and frame number (within batch to be visualized), and applies lines representing steering angle and model-predicted steering angle (if available) to image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[0:2]
    # apply text for frame number and steering angle
    cv2.putText(img, 'frame: ' + str(frame), org=(2, 18), fontFace=font, fontScale=.5, color=(200, 100, 100),
                thickness=1)
    cv2.putText(img, 'angle: ' + str(angle), org=(2, 33), fontFace=font, fontScale=.5, color=(200, 100, 100),
                thickness=1)
    # apply a line representing the steering angle
    cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + angle * w / 4), int(h / 2)), (0, 255, 0), thickness=4)
    if pred_angle is not None:
        cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + pred_angle * w / 4), int(h / 2)), (0, 0, 255), thickness=4)
    return img


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
