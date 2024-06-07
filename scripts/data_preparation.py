# import os
# import cv2
# import numpy as np

# # def load_images(image_paths, size=(256, 256)):
# #     images = []
# #     for image_path in image_paths:
# #         image = cv2.imread(image_path)
# #         image = cv2.resize(image, size)
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         images.append(image)
# #     return np.array(images)

# def display_image(image, title='Image'):
#     cv2.imshow(title, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def process_and_save_images(input_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     file_path = os.path.join(input_dir, 'gray_scale.npy')

#     gray_scale_data = np.load(file_path)


#     # print("mean: ", mean_value)
#     for i, image in enumerate(gray_scale_data):   
#         image = cv2.resize(image, (256, 256))
#         # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         cv2.imwrite(os.path.join(output_dir, f'{i}.jpg'), image)
#         # image = image/255.0
#         # np.save(os.path.join(output_dir, f'{i}.npy'), image)




# train_dir = '../data/raw/train'
# processed_train_dir = '../data/processed/train'
# process_and_save_images(train_dir, processed_train_dir)



########################################################

import os
import cv2
import numpy as np

def resize_images(image_file, size=(256, 256)):
    resized_image = []
    for image in image_file:
        image = cv2.resize(image, size)
        # image = image/255.0
        resized_image.append(image)
    return np.array(resized_image)

def load_grayscale_images(file_path):
    image_array = np.load(file_path)
    resized_images = resize_images(image_array)
    return resized_images

def load_color_images(file_paths):
    color_images = [np.load(file_path) for file_path in file_paths]
    resized_images = [resize_images(np.concatenate(color_images, axis=0))]
    return resized_images

def save_processed_data(grayscale_images, color_images, output_dir):
# def save_processed_data(grayscale_images,  output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    # Save grayscale and color images as separate .npy files
    np.save(os.path.join(output_dir, 'X_train.npy'), grayscale_images)
    np.save(os.path.join(output_dir, 'y_train.npy'), color_images)

# File paths for grayscale and color images
grayscale_file_path = '../data/raw/train/gray_scale.npy'
color_file_paths = ['../data/raw/train/ab/ab1.npy', '../data/raw/train/ab/ab2.npy', '../data/raw/train/ab/ab3.npy']

# Load images
grayscale_images = load_grayscale_images(grayscale_file_path)
color_images = load_color_images(color_file_paths)

# Normalize grayscale images to [0, 1]
# grayscale_images = grayscale_images / 255.0
# color_images = color_images / 255.0

# Output directory for processed data
processed_output_dir = '../data/processed/train/'

# Save processed data
save_processed_data(grayscale_images, color_images, processed_output_dir)
# save_processed_data(grayscale_images,  processed_output_dir)


print(f"Processed data saved to {processed_output_dir}")
