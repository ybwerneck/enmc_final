# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 00:28:09 2023

@author: yanbw
"""

import cv2
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(1, 14):
        time_in_seconds = i
        frame_number = int(frame_rate * time_in_seconds)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(output_folder, f'{i}.png')
            cv2.imwrite(frame_path, frame)
            print(f'Extracted frame {i} from {video_path}')
        else:
            print(f'Error reading frame {i} from {video_path}')

    cap.release()

if __name__ == "__main__":
    video_folder = ""
    output_base_folder = ""

    video_files = ["agua.mp4", "espuma.mp4", "ar.mp4"]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_folder = os.path.join(output_base_folder, os.path.splitext(video_file)[0])

        extract_frames(video_path, output_folder)
