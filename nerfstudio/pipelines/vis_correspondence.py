# DEMO FOR VISUALIZATION OF CORRESPONDENCES
# This script shows how to visualize the correspondences between three images
import matplotlib.pyplot as plt
import numpy as np

def vis_correspondence_three_imgs(image_batch, cam_0_yx_coordinates, projected_coordinates_0_1, projected_coordinates_0_2, H, W):
    image_0_tensor = image_batch["image"][0].cpu().numpy()  # Convert to numpy array
    image_1_tensor = image_batch["image"][1].cpu().numpy()  # Convert to numpy array
    image_2_tensor = image_batch["image"][2].cpu().numpy()  # Convert to numpy array

    # Normalize the pixel values to the range [0, 1]
    first_image_normalized = (image_0_tensor - np.min(image_0_tensor)) / (np.max(image_0_tensor) - np.min(image_0_tensor))
    second_image_normalized = (image_1_tensor - np.min(image_1_tensor)) / (np.max(image_1_tensor) - np.min(image_1_tensor))
    third_image_normalized = (image_2_tensor - np.min(image_2_tensor)) / (np.max(image_2_tensor) - np.min(image_2_tensor))

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    # Plot the first image and its projected coordinates
    axs[0].imshow(first_image_normalized)
    for i, (y, x) in enumerate(cam_0_yx_coordinates):
        axs[0].text(x, y, str(i), color="red", fontsize=10, ha="center", va="center")
    axs[0].set_title("First Image with Cam 0 Coordinates")
    axs[0].axis("off")

    # Plot the second image and its projected coordinates
    axs[1].imshow(second_image_normalized)
    for i, (y, x) in enumerate(projected_coordinates_0_1):
        if 0 <= y < H and 0 <= x < W:
            axs[1].text(x, y, str(i), color="blue", fontsize=10, ha="center", va="center")
    axs[1].set_title("Second Image with Projected Coordinates")
    axs[1].axis("off")

    # Plot the third image and its projected coordinates
    axs[2].imshow(third_image_normalized)
    for i, (y, x) in enumerate(projected_coordinates_0_2):
        if 0 <= y < H and 0 <= x < W:
            axs[2].text(x, y, str(i), color="green", fontsize=10, ha="center", va="center")
    axs[2].set_title("Third Image with Projected Coordinates")
    axs[2].axis("off")

    # Show the plots
    plt.tight_layout()
    plt.show()

#DEMO FOR VISUALIZATION OF DEPTH IMAGES
def vis_depth_images(depths):
    import matplotlib.pyplot as plt
    import numpy as np

    depth_0_tensor = depths[0].cpu().numpy()  # Convert to numpy array
    depth_1_tensor = depths[1].cpu().numpy()  # Convert to numpy array
    depth_2_tensor = depths[2].cpu().numpy()  # Convert to numpy array

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    axs[0].imshow(depth_0_tensor)
    axs[0].set_title("First Depth Image")
    axs[0].axis("off")

    axs[1].imshow(depth_1_tensor)
    axs[1].set_title("Second Depth Image")
    axs[1].axis("off")

    axs[2].imshow(depth_2_tensor)
    axs[2].set_title("Third Depth Image")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()