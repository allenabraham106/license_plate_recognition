import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
from skimage.morphology import closing, opening, square

# from skimage.morphology import remove_small_objects
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca

# inverts the white and the black
license_plate = np.invert(
    cca.plate_like_objects[1]
)  # hardcoded value, can be switched around

# open to remove noise and then close to fill
license_plate = opening(license_plate, square(1))
license_plate = closing(license_plate, square(1))
labeled_plate = measure.label(license_plate)

# show our images
fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

# rough estimates to what a license plate might look like
h, w = license_plate.shape
character_dimensions = (
    # Height is 15 - 80% of the plate
    0.15 * h,
    0.80 * h,
    # Width is 0.5 - 16% of the plate
    0.005 * w,
    0.16 * w,
)

min_height, max_height, min_width, max_width = character_dimensions
character = []
counter = 0
coloumn_list = []

for region in regionprops(labeled_plate):
    y0, x0, y1, x1 = region.bbox
    region_height = y1 - y0
    region_width = x1 - x0
    region_area = region.area
    min_area = 50

    if (
        region_height > min_height
        and region_width > min_width
        and region_height < max_height
        and region_width < max_width
        and region_area > min_area
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

        print(
            f"Character at x={x0}: h={region_height:.1f}, w={region_width:.1f}, area={region_area}"
        )

        # keep track on the arrangement of characters
        coloumn_list.append(x0)

plt.show()
