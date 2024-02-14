import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

# Define a list to hold the coordinates of each ROI
roi_corners = []

# Plot the image
fig, ax = plt.subplots()
ax.imshow(img)

# Callback function to get the coordinates of the rectangular ROI
def onselect(eclick, erelease):
    # Access the global list
    global roi_corners
    # Append the coordinates as a tuple (x1, y1, x2, y2)
    x1, y1, x2, y2 = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)
    roi_corners.append((x1, y1, x2, y2))
    print(f"Rectangle added with coordinates: {roi_corners[-1]}")
    
    # Draw the rectangle on the plot
    ax.add_patch(Rectangle((x1, y1), abs(x2-x1), abs(y2-y1),
                           edgecolor='red', facecolor='none', lw=2))
    fig.canvas.draw()

# Set up the RectangleSelector widget
toggle_selector = RectangleSelector(ax, onselect, useblit=False,
                                    button=[1],  # Left mouse button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)

plt.show()
