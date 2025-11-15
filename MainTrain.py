import cv2;
import os;

image_directory = os.path.dirname(os.path.abspath(__file__))
no_tumour_images=os.listdir(os.path.join(image_directory, 'no'))

print(no_tumour_images)