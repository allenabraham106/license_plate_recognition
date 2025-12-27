# Connected Component Analysis
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

# getting all the connected regions and grouping them
label_image = measure.label(localization.binary_car_image)

# based on research, we are getting the dimensions of possible license plate dimensions
plate_dimensions = (
    0.03 * label_image.shape[0],
    0.25 * label_image.shape[0],
    0.1 * label_image.shape[1],
    0.6 * label_image.shape[1],
)
min_height, max_height, min_width, max_width = plate_dimensions
plate_object_cordinates = []
plate_like_objects = []
fig, ax1 = plt.subplots(1)
ax1.imshow(localization.binary_car_image, cmap="gray")

# regionprops now creates a list of properties for labelled groups
for region in regionprops(label_image):
    if region.area < 50:
        continue  # region area is probably so small that its not a license plate
    # Box Cordinatees
    minRow, minCol, maxRow, maxCol = region.bbox
    box_width = maxCol - minCol
    box_height = maxRow - minRow

    if (
        box_height >= min_height
        and box_height <= max_height
        and box_width >= min_width
        and box_width <= max_height
        and box_height < box_width
    ):
        plate_like_objects.append(
            (localization.binary_car_image[minRow:maxRow, minCol:maxCol])
        )
        plate_object_cordinates.append((minRow, minCol, maxRow, maxCol))
        rectBorder = patches.Rectangle(
            (minCol, minRow),
            maxCol - minCol,
            maxRow - minRow,
            edgecolor="red",
            linewidth=2,
            fill=False,
        )
        ax1.add_patch(
            rectBorder
        )  # drawing a recatangle over the parts of the immage that match our dimensions

plt.show()
