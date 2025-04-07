import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import os  # Import os module to list files
from pathlib import Path  # Import Path for better path handling

#out_dir = Path("C://Users//rhodr//OneDrive//Documents//GitHub//CV_mini_project//test//res//")# Output directory for saving images"

# Initialize global variables
global img, draw
img = Image.new("RGB", (500, 500), "white")  # Default blank canvas
draw = ImageDraw.Draw(img)  # Initialize draw object
last_clicked_point = None 
image_dims = None

def save_image():
    global img
    if img is not None:  # Ensure img is initialized
        try:
            selected_directory = filedialog.askdirectory(title="Select Directory to Save Images")  # Ask user for save directory from the UI
            output_path = os.path.join(selected_directory, 'Image_plus_prompt.png')  # Pathlib ensures proper path handling
            img.save(output_path)  # Save the image with points
            print(f"Image saved successfully at {output_path}!")
        except Exception as e:
            print(f"Error saving image: {e}")
    else:
        print("No image to save!")

def paint(event):
    global img, draw, last_clicked_point, image_dims
    x, y = event.x, event.y  # Get the coordinates of the mouse event
    last_clicked_point = (x, y)  # Store the clicked point
    image_dims = img.size  # Get the dimensions of the image
    canvas.create_rectangle(x, y, x + 3, y + 3, fill="red", outline="red")  # Represent the point on the canvas
    draw.point((x, y), fill="red")  # Draw a single point on the Pillow image


def open_file(event):
    global img, photo, draw
    # Get the selected file from the listbox
    selected_file = file_listbox.get(file_listbox.curselection())
    filepath = os.path.join(directory, selected_file)  # Combine directory path with filename

    # Open the selected image so it is behind the text and update globals
    global img, draw

    img = Image.open(filepath).convert("RGB")  # Open the image and convert to RGB
 #   img.thumbnail((450, 450))  # Resize image to fit canvas
    photo = ImageTk.PhotoImage(img)  # Create a PhotoImage object for Tkinter
    canvas.delete("all")  # Clear the canvas before loading the new image
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)  # Display image on the canvas
    draw = ImageDraw.Draw(img)  # Update the draw object for the new image

# Tkinter UI setup
root = tk.Tk()
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()

# Choose the directory containing images
directory = filedialog.askdirectory(title="Select Image Directory")
files = os.listdir(directory)  # List all files in the directory
image_files = [f for f in files if f.endswith((".jpg", ".jpeg", ".png"))]  # Filter image files

#add text formatted in a box that stays in the foreground to the canvas in the UI with instructions to double click to open image files then click to draw points
canvas.create_rectangle(0, 460, 500, 500, fill="lightgrey", outline="black")  # Create a rectangle for the text background
canvas.create_text(250, 480, text="Double-click to open an image file, then click to draw points.", fill="black", font=("Arial", 12))

# Create a Listbox to show files
file_listbox = tk.Listbox(root, height=10, width=50)
file_listbox.pack()
for file in image_files:
    file_listbox.insert(tk.END, file)  # Add each file to the Listbox

file_listbox.bind("<Double-Button-1>", open_file)  # Open file on double click

# Change button to save and close the UI
save_button = tk.Button(root, text="Save and Close", command=lambda: [save_image(), root.quit()])
save_button.pack()
canvas.bind("<Button-1>", paint)  # Enable point drawing on mouse click



root.mainloop()
print(f'Clicked x and y co-ordinates {last_clicked_point}')
print(f'Image dimensions are {img.size}')





