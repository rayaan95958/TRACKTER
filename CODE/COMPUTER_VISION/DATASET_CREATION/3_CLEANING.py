#Deletes all pictures with no correspondimng annotations.

import os

# Define the directory path
annotated_dir = r'C:\Users\satar\OneDrive\Desktop\TRACKTER\DATA\DATASET_ANNOTATED'

def delete_images_without_annotations():
    # Get all image and annotation files
    img_files = [f for f in os.listdir(annotated_dir) if f.endswith('.png')]
    txt_files = [f for f in os.listdir(annotated_dir) if f.endswith('.txt')]

    # Create a set of annotation filenames without extension
    txt_basenames = {os.path.splitext(f)[0] for f in txt_files}

    # Delete images without corresponding annotations
    for img_file in img_files:
        img_basename = os.path.splitext(img_file)[0]
        if img_basename not in txt_basenames:
            img_path = os.path.join(annotated_dir, img_file)
            os.remove(img_path)
            print(f'Deleted {img_path}')

# Execute the function
delete_images_without_annotations()

print("Deletion of images without annotations completed.")
