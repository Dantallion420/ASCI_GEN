# ASCI_GENASCII Art Generator
This Python script allows you to convert images or videos into ASCII art. The generated ASCII art can be saved as images or compiled into a video.

Prerequisites
Python 3.x
Libraries: PIL (Python Imaging Library), OpenCV (cv2), imageio, numpy, tqdm
Usage
Clone or download the repository to your local machine.
Navigate to the directory containing the script.
Run the script by executing the following command:
bash
Copy code
python ascii_art_generator.py
Follow the prompts to provide input parameters such as the path to the image or video file, maximum height of the ASCII art, character width to height ratio, and maximum number of frames to process.

Code Explanation
The script consists of several functions:

calculate_threshold(roi): Calculates the threshold based on the average intensity of pixels in a Region of Interest (ROI).
adjust_saturation_contrast(image, saturation_factor, contrast_factor): Adjusts the saturation and contrast of an image.
generate_ascii_from_roi(roi, max_height, aspect_ratio): Generates ASCII art from a Region of Interest.
generate_ascii(image_path, max_height, aspect_ratio, rois=None): Generates ASCII art from an image or video frame.
main(): The main function of the script. Accepts user input, processes frames from the input file, detects objects using YOLOv3, generates ASCII art, and compiles ASCII frames into a video.
Output
The generated ASCII art frames are saved as images in the "./[filename]_Images" directory and compiled into a video named "ascii_art.mp4" in the "./[filename]_AsciiFiles" directory.

Note
For video input, the script processes a limited number of frames to enhance performance. You can adjust this limit as needed.
Ensure that you have the necessary weights and configuration files for the YOLOv3 object detector (yolov3.weights, yolov3.cfg, coco.names) in the script directory for object detection to work properly.
