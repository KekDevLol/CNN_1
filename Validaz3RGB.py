import os
import cv2

def convert_to_rgb(input_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                if image is not None:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    new_image_path = os.path.join(root, f"{file}")
                    cv2.imwrite(new_image_path, rgb_image)
                    print(f"Converted {image_path} to {new_image_path}")
                else:
                    print(f"Failed to read {image_path}")

# Пример использования
input_folder = 'validation'
convert_to_rgb(input_folder)