import sys
import os
# Add the folder to the Python path (Rhodri - I need this line to run the file but others can remove it)
sys.path.append(os.path.abspath(r"C://Users//rhodr//OneDrive//Documents//GitHub//CV_mini_project//src//models"))
import torch
from torchvision.transforms.functional import to_pil_image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import os
import resizing
from clip_seg import Clip

import argparse
import sys
import os



def get_args():
    parser = argparse.ArgumentParser(description="File path for model weights (string).")
    parser.add_argument("--weights_path", type=str, required=True, 
                        help="The file path to the model weights (e.g. weights.pth)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()  
    
    # Initialise model and procload weights
    model = Clip()
    try:
        model.load_state_dict(torch.load(args.weights_path))
        print(f"Model weights loaded successfully from {args.weights_path}!")
    except Exception as e:
        print(f"Error loading model weights: {e}")


# Initialize global variables
img = Image.new("RGB", (500, 500), "white")  # Default blank canvas
draw = ImageDraw.Draw(img)  
last_clicked_point = None
image_dims = None
img_resize = None
rgb_image_resize = None

def save_image():
    ''' Save the image as Image_plus_prompt.png with points drawn on it in the directory selected by the user '''
    global img_resize, rgb_image_resize
    if img is not None:  # Ensure img is initialized
        try:
            selected_directory = filedialog.askdirectory(title="Select Directory to Save Images")  # Ask user for save directory from the UI
            output_path = os.path.join(selected_directory, 'Image_plus_prompt.png')  # Pathlib ensures proper path handling
            img_resize.save(output_path)  # Save the image with points
            output_path_masked = os.path.join(selected_directory, 'Image_plus_mask.png')
            rgb_image_resize.save(output_path_masked)  # Save the image with points
            print(f"Image saved successfully at {output_path}!")
        except Exception as e:
            print(f"Error saving image: {e}")
    else:
        print("No image to save!")

def paint(event):
    ''' Draw a red point on the image at the clicked location and stores the x,y co-ordinate '''
    global img, draw, last_clicked_point, image_dims
    x, y = event.x, event.y  
    last_clicked_point = (x, y)  
    image_dims = img.size  
    canvas.create_rectangle(x, y, x + 5, y + 5, fill="red", outline="red")  # Represent the point on the canvas
    draw.point((x, y), fill="red")  

def open_file(event):
    ''' Open the image file and display it on the canvas '''
    global img, photo, draw
    # Get the selected file from the listbox
    selected_file = file_listbox.get(file_listbox.curselection())
    filepath = os.path.join(directory, selected_file)  # Combine directory path with filename

    img = Image.open(filepath).convert("RGB")  
    photo = ImageTk.PhotoImage(img)  
    canvas.delete("all")  # Clear the canvas before displaying the image
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)  
    draw = ImageDraw.Draw(img)

def process_images():
    ''' Process the image with the model and prepare img_resize and rgb_image_resize for display '''
    global img_resize, rgb_image_resize  # Declare as global to use in other functions
 #   model = Clip()
 #   model.load_state_dict(torch.load(r"C://Users//rhodr//Documents//CV_Dataset//CLIP_prompt.pth"))  # Load the model weights
    img_resize = resizing.resize_image(img, (352, 352))  # Resize the image to the model's input size
    y = model.forward(img_resize, 'Cat or dog')  # Forward pass through the model

    y_pred_classes = torch.argmax(torch.softmax(y, dim=1), dim=1)  # (N, W, H)
    y_pred_classes = y_pred_classes[0]

    H, W = y_pred_classes.shape
    # Create an empty RGB image (H, W, 3)
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)

    # Define class-to-color mapping
    rgb_image[y_pred_classes == 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)    # Black (Class 0)
    rgb_image[y_pred_classes == 1] = torch.tensor([0,255, 0], dtype=torch.uint8)  # Red (Class 1)
    rgb_image[y_pred_classes == 2] = torch.tensor([0, 0, 0], dtype=torch.uint8)  # Green (Class 2)
    rgb_image = to_pil_image(rgb_image.permute(2, 0, 1)).convert("RGB")

    rgb_image_resize = rgb_image.resize((img_resize.size[0], img_resize.size[1]), Image.NEAREST)  # Resize the predicted image to match the original image size

    rgb_image_resize = Image.blend(img_resize, rgb_image_resize, alpha=0.33)  # Blend the original image with the predicted image

def display_images(img_1, mask_1):
    ''' Clear the canvas and display the resized image alongside the dame image with the predicted mask overlay'''
    global canvas
    # Clear all elements on the canvas (including the background image)
    canvas.delete("all")
    
    # Convert the images to Tkinter-compatible formats
    img_photo = ImageTk.PhotoImage(img_1)
    mask_photo = ImageTk.PhotoImage(mask_1)
    
    # Display the images side by side on the canvas
    canvas.create_image(50, 50, anchor=tk.NW, image=img_photo)  # Left: img_resize
    canvas.create_image(450, 50, anchor=tk.NW, image=mask_photo)  # Right: rgb_image_resize

    # Store references to the images to prevent garbage collection
    canvas.img_photo = img_photo
    canvas.mask_photo = mask_photo

def save_and_display_images():
    ''' Save the image and display the resized images side by side '''
    process_images()  # Process the image and prepare for display
    save_image()  # Save the current image with points
    display_images(img_resize, rgb_image_resize)  # Replace canvas content with the side-by-side images
    print("Images displayed on canvas successfully!")

# Tkinter UI setup
root = tk.Tk()
canvas = tk.Canvas(root, width=1000, height=500)
canvas.pack()

# Choose the directory containing images and list all the image files for the user to select from
directory = filedialog.askdirectory(title="Select Image Directory")
files = os.listdir(directory)  
image_files = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]  

# Create a Listbox to show files
file_listbox = tk.Listbox(root, height=10, width=50)
file_listbox.pack()
for file in image_files:
    file_listbox.insert(tk.END, file)  

file_listbox.bind("<Double-Button-1>", open_file)  # Open file on double click

# Add instructions to double click to open image files then click to draw points
canvas.create_rectangle(20, 460, 980, 500, fill="lightgrey", outline="black")  
canvas.create_text(500, 480, text="Double-click to open the selected image, click to draw point and 'Save' to save and display images", 
                   fill="black", font=("Arial", 12))

# Add Save and Close buttons
save_button = tk.Button(root, text="Save and Run", command=save_and_display_images)
save_button.pack()

close_button = tk.Button(root, text="Close", command=root.quit)
close_button.pack()

canvas.bind("<Button-1>", paint)  # Enable point drawing on mouse click
canvas.tag_raise("info")  # Raise the info text to the top layer

root.mainloop()