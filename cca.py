# Connected Component Analysis
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

# getting all the connected regions and grouping them
label_image = measure.label(localization.binary_car_image)
fig, ax1 = plt.subplots(1)
ax1.imshow(localization.binary_car_image, cmap="gray")

# regionprops now creates a list of properties for labelled groups
for region in regionprops(label_image):
    if region.area < 50:
        continue  # region area is probably so small that its not a license plate
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle(
        (minCol, minRow),
        maxCol - minCol,
        maxRow - minRow,
        edgecolor="red",
        linewidth=2,
        fill=False,
    )
    ax1.add_patch(rectBorder)  # drawing a rectangle over the matches

plt.show()
