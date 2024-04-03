from PIL import Image, ImageDraw , ImageFilter
import cv2
import os
import imageio
import numpy as np
from tqdm import tqdm
import random
def calculate_threshold(roi):
    # Calculate the average intensity of pixels in the ROI
    average_intensity = np.mean(roi)
    # Adjust the threshold based on the average intensity or other characteristics of the ROI
    threshold = 0.6 * average_intensity  # Adjust as needed
    return threshold
def adjust_saturation_contrast(image, saturation_factor=1.0, contrast_factor=1.0):
    # Convert image to BGR format (OpenCV format)
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert BGR image to HSV format
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Adjust saturation
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 255)

    # Adjust contrast
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] * contrast_factor, 0, 255)

    # Convert HSV image back to BGR format
    modified_bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Convert BGR image back to RGB format
    modified_rgb_image = cv2.cvtColor(modified_bgr_image, cv2.COLOR_BGR2RGB)

    # Convert modified image to PIL Image format
    modified_image = Image.fromarray(modified_rgb_image)

    return modified_image

def generate_ascii_from_roi(roi, max_height, aspect_ratio):
    # Dynamically calculate threshold based on ROI characteristics
    threshold = calculate_threshold(roi)
    # Adjust other parameters as needed
    block_size = 3
    
    # Convert ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 

    # Resize ROI while maintaining aspect ratio
    new_width = int(roi_gray.shape[1] * aspect_ratio)
    roi_resized = cv2.resize(roi_gray, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)

    ascii_art = ""
    for y in range(roi_resized.shape[0]):
        for x in range(roi_resized.shape[1]):
            # Check if pixel intensity is below dynamically calculated threshold
            if roi_resized[y, x] < threshold:
                # Represent background with dark pixels
                ascii_art += " "  
            else:
                # Represent object with blinking 0s and 1s
                if random.random() < 0.9:
                    ascii_art += "0"  
                else:
                    ascii_art += "1"  
        ascii_art += "\n"

def generate_ascii(image_path, max_height, aspect_ratio, rois=None):
    # Define binary characters for ASCII representation
    binary_chars = "01"
    threshold = 128
    block_size = 3
       # Calculate threshold
    
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert("L")
     # Adjust saturation and contrast
    
    # Resize image while maintaining aspect ratio
    new_width = int(img.size[1] * aspect_ratio)
    img = img.resize((new_width, max_height), Image.LANCZOS)  # Using LANCZOS for antialiasing

    ascii_art = ""
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            # Get pixel intensity value
            pixel_intensity = img.getpixel((x, y))
            # Check if pixel intensity is above threshold
            if pixel_intensity < threshold:
                # Represent background with dark pixels
                ascii_art += " "  
            else:
                # Represent object with blinking 0s and 1s
                if random.random() < 0.8:
                    ascii_art += "0"  
                else:
                    ascii_art += "1"  
        ascii_art += "\n"
    
    # Process ROIs and append their ASCII art
    if rois is not None:
        for roi in rois:
            ascii_art += generate_ascii_from_roi(roi, max_height, aspect_ratio)
    
    return ascii_art
      
def main():
    # Accept input parameters from the user
    input_file = input("Enter the path to the image or video file: ")
    max_height = int(input("Enter the maximum height of the ASCII art: "))
    aspect_ratio = float(input("Enter the character width to height ratio - from 0.1 to 5: "))
    frame_limit = int(input("Enter the maximum number of frames to process - min 100 : "))
    # Create output directories if needed
    output_image_dir = "./%s_Images" % os.path.splitext(os.path.basename(input_file))[0]
    output_ascii_dir = "./%s_AsciiFiles" % os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_ascii_dir, exist_ok=True)
    # Initialize background subtractor
    
        # Capture frames from video or process single image
    ascii_frames = []
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.flv' ,'.jpeg','.png')):
        vidcap = cv2.VideoCapture(input_file)
        total_frames = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_limit)  # Limit total frames to process
        pbar = tqdm(total=total_frames, desc="Processing Frames")
        success, image = vidcap.read()
        count = 0
        while success and count < frame_limit:  # Check if count exceeds frame limit
            frame_path = os.path.join(output_image_dir, "frame%d.jpg" % count)
            cv2.imwrite(frame_path, image)
            ascii_art = generate_ascii(frame_path, max_height, aspect_ratio)
            if not ascii_art:
                print("Failed to generate ASCII art for frame %d" % count)
                continue
            ascii_frames.append(ascii_art)
            pbar.update(1)
            success, image = vidcap.read()
            count += 1
        pbar.close()
        print("Done processing frames")
    else:
        # Process single image
        ascii_art = generate_ascii(input_file, max_height, aspect_ratio)
        if not ascii_art:
            print("Failed to generate ASCII art.")
            return
        with open(os.path.join(output_ascii_dir, "ascii_art.jpeg"), "w") as f:
            f.write(ascii_art)
        ascii_frames.append(ascii_art)
        print("ASCII art generated successfully")
        # Initialize object detector (YOLOv3)
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        ascii_frames = []
        pbar = tqdm(total=total_frames, desc="Processing Frames")
        count = 0
        while count < frame_limit:
            ret, frame = vidcap.read()
            if not ret:
                break
            # Get ROI around detected object
            roi = frame[y:y + h, x:x + w]
            
            # Convert ROI to ASCII art with dynamically calculated threshold
            ascii_roi = generate_ascii_from_roi(roi, max_height, aspect_ratio)
            ascii_frame += ascii_roi
            # Detect objects in the frame
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            
                # Detect objects in the frame
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
                    # Process detected objects
            # Process detected objects
            ascii_frame = ""
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:  # Minimum confidence threshold
                        # Extract object bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        if classes[class_id] == 'person':
                        # Get ROI around detected person
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            roi = frame[y:y + h, x:x + w]
                            
                            # Convert ROI to ASCII art
                            ascii_frame = generate_ascii_from_roi(roi, max_height, aspect_ratio)
                            ascii_frames.append(ascii_frame)
                            
                        # Convert ROI to ASCII art
                        ascii_frame = generate_ascii(roi, max_height, aspect_ratio)
                        ascii_frames.append(ascii_frame)

            # Write frame image to disk (optional)
            frame_path = os.path.join(output_image_dir, "frame%d.jpg" % count)
            cv2.imwrite(frame_path, frame)
            
            pbar.update(1)
            count += 1
        
    pbar.close()
    vidcap.release()
    # Save ASCII frames as a video (mp4)
    output_video_path = os.path.join(output_ascii_dir, "ascii_art.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (1000, 1000))
    for ascii_frame in tqdm(ascii_frames, desc="Generating Video"):
        # Create image from ASCII art
        img = Image.new("RGB", (1000, 1000), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ascii_frame, fill=(255, 255, 255))
        # Convert image to cv2 format
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video_writer.write(img_array)
    video_writer.release()
    print("ASCII frames saved as video:", output_video_path)
    print("Processing Complete!")

if __name__ == "__main__":
    main()
