from pathlib import Path
from torchvision.transforms import v2, InterpolationMode
from PIL import Image
import numpy as np
import cv2
import numpy as np

src_dir = "C://Users//rhodr//Documents//CV_Dataset//TrainVal//train//"
dest_dir = "C://Users//rhodr//Documents//CV_Dataset//TrainVal//train_centre//"

src_dir_label = "C://Users//rhodr//Documents//CV_Dataset//TrainVal//label//"
dest_dir_label = "C://Users//rhodr//Documents//CV_Dataset//TrainVal//label_centre//"

#src_dir = "C://Users//rhodr//OneDrive//Documents//GitHub//CV_mini_project//test//res_out//"
#dest_dir = "C://Users//rhodr//OneDrive//Documents//GitHub//CV_mini_project//test//res_out//train_centre//"
#src_dir_label = "C://Users//rhodr//OneDrive//Documents//GitHub//CV_mini_project//test//res_out//label//"
#dest_dir_label = "C://Users//rhodr//OneDrive//Documents//GitHub//CV_mini_project//test//res_out//label_centre//"
dim = (256, 256)



def mask_centre(img, label, filename: str, list_no_valid_centroid, list_centroid_not_in_mask) -> np.ndarray:
    '''
    Calculate the centroid of a binary mask and draws it on the image and mask.

    Args:
    -----   
        img (np.ndarray): image to mark with centroid
        label (np.ndarray): mask to calculate centroid from and mark
        filename (str): name of the image file for logging

    Returns:
    --------
        output_img (np.ndarray): image with centroid marked
        output_label (np.ndarray): mask with centroid marked
        centre (tuple): coordinates of the centroid

    '''

    def calculate_centre(mask):
        '''
        Calculate centre of mask from moments

        Args:
        -----
            mask (np.ndarray): mask to calculate centroid from
        
        Returns:
        --------
            centre (tuple): coordinates of the centroid where it exists

        '''
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] != 0:
            centre_x = int(M["m10"] / M["m00"])
            centre_y = int(M["m01"] / M["m00"])
            return centre_x, centre_y
        else:
            return None

    # Convert the label image to grayscale and create a binary mask
    lab_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(lab_gray, 1, 255, cv2.THRESH_BINARY)
    # Find the centroid of the binary mask
    centre = calculate_centre(binary_mask)
    if centre is not None:
        # Check if the centroid is within the mask (i.e. is a red or green pixel)
        if label[centre[1], centre[0]][1] == 128 or label[centre[1], centre[0]][2] == 128:
            # draw a point on a blank image at the centroid position (red for cats and green for dogs)
            img_zeros = np.zeros_like(img)
            if label[centre[1], centre[0]][1] == 128:
                img_pt = cv2.circle(img_zeros, centre, radius=5, color=(0, 255, 0), thickness=-1)
            elif label[centre[1], centre[0]][2] == 128:
                img_pt = cv2.circle(img_zeros, centre, radius=5, color=(0, 0, 255), thickness=-1)
            #run a guassian filter on the single point so that it has a radius of 9 pixels
            img_pt_blur = cv2.GaussianBlur(img_pt, (9, 9), 0)
            #combine the original image and label with the blurred point (weight the image down by 5% so the pixels are visible on white images)
            output_img = cv2.addWeighted(img, .95, img_pt_blur, 1, 0)
            output_label = cv2.addWeighted(label, .95, img_pt_blur, 1, 0)
            #clip the image and label so no rgb value is above 255
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            output_label = np.clip(output_label, 0, 255).astype(np.uint8)



        else:
            # if the centroid is not within the mask, return the original image and label
            print(f"{filename} has no valid mask found at the centroid.")
            list_centroid_not_in_mask.append(filename) 
            output_label = label.copy()
            output_img = img.copy()

    else:
        #if no centroid is found, return the original image and label
        print(f"{filename} has no valid centriod in mask {np.unique(label)}")
        list_no_valid_centroid.append(filename)
        output_label = label.copy()
        output_img = img.copy()
    return output_img, output_label, centre, list_no_valid_centroid, list_centroid_not_in_mask

def mask_centre_directory(src_dir: Path, dest_dir: Path, src_dir_label: Path, dest_dir_label: Path, dim: tuple [int,int] ,print_progress: bool = True):
    """
    Calculate the centroid of a binary mask and draw it on the image.

    Args:
    -----
        src_dir (Pathlib.Path): path where source images are located.
        dest_dir (Pathlib.Path): path where resized images are saved.
        src_dir_lab (Pathlib.Path): path where source label images are located.
        dest_dir_lab (Pathlib.Path): path where resized label images are saved.
        dim (tuple): dimensions of the image
        print_progress (bool): whether to print progress or not

    Returns:
    --------
        None
    """
    accepted_file_types = {".jpg", ".jpeg", ".png"}
    dir_files = list(src_dir_label.iterdir())
    nbr_images = len([f for f in dir_files if f.suffix.lower() in accepted_file_types])
    count = 0
    list_no_valid_centroid=list()
    list_centroid_not_in_mask=list()

    for label_path in dir_files:
        if label_path.suffix.lower() in accepted_file_types:
            label = cv2.imread(str(label_path))
            if label is None:
                print(f"Error loading label: {label_path}")
                continue
            img_path = src_dir / f"{Path(label_path).stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error loading image: {img_path}")
                continue
          
            output_img, output_label, centre, list_no_valid_centroid, list_centroid_not_in_mask = mask_centre(img, label, label_path, list_no_valid_centroid, list_centroid_not_in_mask)

            if output_label is not None:
                output_path_label = dest_dir_label / label_path.name
                cv2.imwrite(str(output_path_label), output_label)
                output_path_img = dest_dir / img_path.name
                cv2.imwrite(str(output_path_img), output_img)
                count += 1
                if print_progress:
                    print(f"Processed {count}/{nbr_images}: {img_path.name}")
            else:
                print(f"Skipping {img_path.name}: No valid mask found.")
    print(f"List of images with no valid centroid so no point added: {len(list_no_valid_centroid)}")
    print(f"List of images with centroid not in mask so no point added : {len(list_centroid_not_in_mask)}")
    print(f"Processing complete: {count}/{nbr_images} images processed.")
    

if __name__ == "__main__":
    mask_centre_directory(src_dir=Path(src_dir), dest_dir=Path(dest_dir), src_dir_label=Path(src_dir_label), dest_dir_label=Path(dest_dir_label), dim =dim, print_progress=True)







