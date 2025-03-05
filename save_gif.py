import os
import imageio

import os
import imageio

def save_gif_from_frames(frames_folder="dimt_rrt_frames", output_gif="dimt_rrt.gif", frame_duration=300, pause_duration=3000):
    """
    Create a looping GIF from frame images with a pause at the end.

    Args:
        frames_folder (str): Folder containing the frame images.
        output_gif (str): Output GIF file name.
        frame_duration (float): Duration for each frame (in seconds).
        pause_duration (float): Duration to pause on the last frame (in seconds).
    """
    images = []
    frame_files = sorted(os.listdir(frames_folder))
    for frame_file in frame_files:
        if frame_file.endswith(".png"):
            frame_path = os.path.join(frames_folder, frame_file)
            images.append(imageio.imread(frame_path))
    
    # All frames use frame_duration except the last one uses pause_duration.
    durations = [frame_duration] * (len(images) - 1) + [pause_duration]
    imageio.mimsave(output_gif, images, duration=durations, loop=0)

if __name__ == '__main__':
    save_gif_from_frames()