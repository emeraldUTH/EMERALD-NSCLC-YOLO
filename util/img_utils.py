import pandas as pd
import os
import shutil
import random
from PIL import Image, ImageDraw, ImageFont

# Load images from a directory
ct = f"F:/nsclc/Test1.v1i.folder"
pet = f"F:/nsclc/Test2.v1i.folder"

root_fs = pet
folder_src = f'{root_fs}/whole' 

def img_rename(image_directory):
    # Get the list of files in the folder
    file_list = os.listdir(image_directory)
    # Iterate over the files and rename the images
    for filename in file_list:
        if filename.endswith(".jpg"):
            new_filename = filename[:filename.index("png") - 1] + ".jpg"  # Keep the part before "png", remove '_' with -1 and add ".jpg" extension
            old_path = os.path.join(image_directory, filename)
            new_path = os.path.join(image_directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")

def img_cp(folder_dst):
    global folder_src
    # Get the list of files in folder "a"
    files_dst = os.listdir(folder_dst)

    # Iterate over the files in folder "a"
    for filename in files_dst:
        file_dst_path = os.path.join(folder_dst, filename)
        file_src_path = os.path.join(folder_src, filename)
        
        # Check if the file exists in folder "b"
        if os.path.isfile(file_src_path):
            shutil.copyfile(file_src_path, file_dst_path)
            print(f"File '{filename}' copied from folder {folder_src} to folder {folder_dst}")
        else:
            print(f"File '{filename}' not found in folder {folder_src}")

def create_label(folder):
    files = os.listdir(folder)
    content = os.path.basename(os.path.normpath(folder))

    # Iterate over the files in folder
    for filename in files:
        file_path = filename[:filename.index("jpg")] + "txt"
        with open(file_path, "w") as file:
            file.write(content)

def change_file_ctx(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                content = file.read()
            
            if content.strip() == "benign":
                content = "0"
                with open(file_path, "w") as file:
                    file.write(content)
                    print(f"Content replaced in file: {file_name}")
            elif content.strip() == "malignant":
                content = "1"
                with open(file_path, "w") as file:
                    file.write(content)
                    print(f"Content replaced in file: {file_name}")

# cat_mov_img reads an excel spredsheet containing ID and label columns and moves images from root dir
# to child dirs according to label
def cat_mov_img(folder_path):
    # Read the Excel file
    df = pd.read_excel(f'{folder_path}/labels.xlsx')

    # Iterate through the DataFrame and move files based on ID and LABEL
    for index, row in df.iterrows():
        image_id = row['ID']
        label = row['LABEL']
        image_filename = f"{image_id}_ct.png"
        # image_filename = f"{image_id}_pet.png"

        # Check the label and move the image to the appropriate folder
        if label == 'Benign':
            source_path = os.path.join(f'{folder_path}', image_filename)
            destination_path = os.path.join(f'{folder_path}/benign', image_filename)
            shutil.move(source_path, destination_path)
            print(f"Moved {image_filename} to 'benign' folder.")
        elif label == 'Malignant':
            source_path = os.path.join(f'{folder_path}', image_filename)
            destination_path = os.path.join(f'{folder_path}/malignant', image_filename)
            shutil.move(source_path, destination_path)
            print(f"Moved {image_filename} to 'malignant' folder.")
        else:
            print(f"Invalid label for {image_filename}: {label}")

    print("Finished moving files.")

# ready_dataset resizes images to 240x240, adds to each image a label given as arg and organizes
#labeled images to train/valid/test folders randomly on a ratio of 0.7/0.2/0.1
def ready_dataset(folder_path, label):
    # Define input and output directories
    input_folder = folder_path
    output_folder = folder_path

    # Get user input for the label
    # user_label = input("Enter the label to be added to the images: ")
    user_label = label

    # Create subfolders if they don't exist
    for folder in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

    # Organize images into train, valid, and test folders
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            # Open the image
            img = Image.open(img_path)
            # Resize the image to 240x240 pixels
            img = img.resize((240, 240))

            label_image = img
            # # Add user-provided label to the image
            # label_image = Image.new('RGB', (img.width, img.height + 30), color='white')  # Create a canvas with extra space for label
            # label_image.paste(img, (0, 0))  # Paste the resized image onto the canvas
            # # Add label text at the bottom of the image
            # label_draw = ImageDraw.Draw(label_image)
            # label_font = ImageFont.load_default()  # You can also load a specific font if needed
            # label_draw.text((10, img.height), f'Label: {user_label}', fill='black', font=label_font)

            # Save the image with label
            output_path = os.path.join(output_folder, f'{filename}')
            label_image.save(output_path)

    # Organize images into train, valid, and test folders
    labeled_images = [filename for filename in os.listdir(output_folder) if filename.endswith('.jpg') or filename.endswith('.png')]
    random.shuffle(labeled_images)

    train_ratio = 0.7
    valid_ratio = 0.2
    test_ratio = 0.1

    total_images = len(labeled_images)
    train_split = int(train_ratio * total_images)
    valid_split = int((train_ratio + valid_ratio) * total_images)

    for i, filename in enumerate(labeled_images):
        src_path = os.path.join(output_folder, filename)
        if i < train_split:
            dst_folder = 'train'
        elif i < valid_split:
            dst_folder = 'valid'
        else:
            dst_folder = 'test'
        dst_path = os.path.join(output_folder, dst_folder, filename)
        shutil.move(src_path, dst_path)

    print("Labels added to images and organized into train, valid, and test folders.")


if __name__ == '__main__':
    # dirs = [f'{root_fs}/test/benign', f'{root_fs}/test/malignant', f'{root_fs}/train/benign', f'{root_fs}/train/malignant', f'{root_fs}/valid/benign', f'{root_fs}/valid/malignant']
    # for dir in dirs:
    #     img_rename(dir)
    #     img_cp(dir)
    #     # create_label(dir)
    # cat_mov_img(f"F:/nsclc/Test3.v1/ct")
    ready_dataset(f"F:/nsclc/Test3.v1/pet/benign", "benign")
    ready_dataset(f"F:/nsclc/Test3.v1/pet/malignant", "malignant")
