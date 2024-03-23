import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np




floodnet_palette = [
           190, 153, 153,
            250, 170, 30,
           220, 220, 0,
           107, 142, 35,
           102, 102, 156,
           152, 251, 152,
           119, 11, 32,
           244, 35, 232,
           220, 20, 60,
           52, 83, 84,
          ]


def show_results(images, imgIds, gts, preds, save_dir):
    """
    Apply a color map to a segmentation result.


    :return: A 3D NumPy array representing the colorized segmentation result.
    """
    for i, (img_id, pred, gt, img) in enumerate(zip(imgIds, preds, gts, images)):
        filename = os.path.join(save_dir, f"{img_id}.png")
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        img = np.squeeze(img)

        mask_pred = colorize_mask(pred, floodnet_palette)
        mask_gt = colorize_mask(gt, floodnet_palette)

        img = ((img - img.min()) / (img.max() - img.min()))
        img = (img * 255).astype(np.uint8)

        img = np.transpose(img, (1, 2, 0))
        concatenated_array = np.hstack((img, mask_gt, mask_pred))
        concatenated_image = Image.fromarray(concatenated_array)
        concatenated_image.save(filename)

def colorize_mask(mask, palette):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def visualize_palette(colors, label_names):
    """
    Visualizes a color palette.

    :param colors: A list of colors, where each color is represented as a tuple
                   of three integers (R, G, B) in the range of 0 to 255.
    """
    # Create a figure and a subplot
    fig, ax = plt.subplots(figsize=(10, 2))
    plt.subplots_adjust(bottom=0.65)
    # Create a simple matrix (1 row, N columns) where N is the number of colors
    # Then display each color in the palette as a column in this matrix
    palette = np.array([colors])

    # Display the palette without axis and with 'nearest' interpolation
    ax.imshow(palette)
    # ax.set_axis_off()
    ax.yaxis.set_visible(False)
    ax.set_xticks(np.arange(len(label_names)) + 0.5)
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    plt.savefig("palette.png")

if __name__ == '__main__':
    label_names = ["Background", "Building_Flooded", "Building_Non-Flooded", "Road_Flooded", \
                  "Road_Non-Flooded", "Water", "Tree", "Vehicle", "Pool", "Grass"]
    colors = [
           (190, 153, 153),
            (250, 170, 30),
           (220, 220, 0),
           (107, 142, 35),
           (102, 102, 156),
           (152, 251, 152),
           (119, 11, 32),
           (244, 35, 232),
           (220, 20, 60),
           (52, 83, 84),
          ]
    visualize_palette(colors, label_names)