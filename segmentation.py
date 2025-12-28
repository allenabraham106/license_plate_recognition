import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca

# inverts the white and the black
license_plate = np.invert(
    cca.plate_like_objects[2]
)  # hardcoded value, can be switched around
labeled_plate = measure.label(license_plate)

# show our images
fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

# rough estimates to what a license plate might look like
character_dimensions = (
    # Height is 35 - 60% of the plate
    0.35 * license_plate[0],
    0.60 * license_plate[0],
    # Width is 5 - 15% of the plate
    0.05 * license_plate[1],
    0.15 * license_plate[1],
)

min_height, max_height, min_width, max_width = character_dimensions
character = []
counter = 0
coloumn_list = []

for region in regionprops(labeled_plate):
    y0, y1, x0, x1 = region.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if (
        region_height > min_height
        and region_width > min_width
        and region_height < max_height
        and region_width < max_width
    ):
        roi = license_plate[y0:y1, x0:x1]

        # drawing rectangles over our characters
        rect_border = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False
        )
        ax1.add_patch(rect_border)

        # resize character for character recognition
        resized_characters = resize(roi, (20, 20))
        character.append(resized_characters)

        # keep track on the arrangement of characters
        coloumn_list.append[x0]

plt.show()
