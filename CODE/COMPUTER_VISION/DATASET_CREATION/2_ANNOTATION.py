import cv2
import os
import matplotlib.pyplot as plt

# Directory containing unannotated images
base_image_dir = 'C:\\Users\\satar\\OneDrive\\Desktop\\TRACKTER\\DATA\\DATASET_UNANNOTATED\\train'
# Directory to save annotated images and annotations
annotation_dir = 'C:\\Users\\satar\\OneDrive\\Desktop\\TRACKTER\\DATA\\DATASET_ANNOTATED'
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# Class name
class_name = 'product'

# Key Functionality:
# r - Reset the image (clear all boxes)
# a - Save the annotated image and annotation file, then move to the next image
# q - Quit the program
# f - Move to the next image without saving
# n - Move to the first image of the next directory

# Function to draw bounding box and save annotations
def annotate_image(image_path, output_image_path, annotation_file_path):
    image = cv2.imread(image_path)
    clone = image.copy()
    boxes = []

    def draw_rectangle(event, x, y, flags, param):
        nonlocal boxes, image, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            boxes.append((x, y))
            cv2.rectangle(image, boxes[-2], boxes[-1], (0, 255, 0), 2)

    # Using matplotlib to display image
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')  # Turn off axis

    # Adjust figure size to fit the image
    height, width, _ = image.shape
    fig.set_size_inches(width / 100, height / 100, forward=True)

    def onclick(event):
        nonlocal boxes
        if event.button == 1:  # Left mouse button
            boxes.append((int(event.xdata), int(event.ydata)))
            if len(boxes) % 2 == 0:
                ax.add_patch(plt.Rectangle(boxes[-2], boxes[-1][0] - boxes[-2][0], boxes[-1][1] - boxes[-2][1], edgecolor='g', facecolor='none'))
                fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    def on_key(event):
        nonlocal boxes, image, fig
        if event.key == 'r':
            image = clone.copy()
            boxes = []
            ax.clear()
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            fig.canvas.draw()
        elif event.key == 'a':
            # Save annotated image
            cv2.imwrite(output_image_path, image)

            # Save annotations in YOLO format
            if len(boxes) % 2 == 0:
                height, width, _ = image.shape
                with open(annotation_file_path, 'w') as f:
                    for i in range(0, len(boxes), 2):
                        x_min, y_min = boxes[i]
                        x_max, y_max = boxes[i+1]
                        x_center = (x_min + x_max) / 2 / width
                        y_center = (y_min + y_max) / 2 / height
                        bbox_width = (x_max - x_min) / width
                        bbox_height = (y_max - y_min) / height
                        f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")
            plt.close(fig)
        elif event.key == 'f':
            plt.close(fig)
        elif event.key == 'n':
            plt.close(fig)
        elif event.key == 'q':
            plt.close(fig)
            exit()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Counter for naming output files
counter = 1
directories = []
for root, dirs, files in os.walk(base_image_dir):
    directories.extend([os.path.join(root, d) for d in dirs])

# Iterate over images in nested directories and annotate
for directory in directories:
    images = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
    images.sort()  # Ensure consistent order

    for file in images:
        image_path = os.path.join(directory, file)
        output_image_path = os.path.join(annotation_dir, f'{counter}.png')
        annotation_file_path = os.path.join(annotation_dir, f'{counter}.txt')
        annotate_image(image_path, output_image_path, annotation_file_path)
        counter += 1