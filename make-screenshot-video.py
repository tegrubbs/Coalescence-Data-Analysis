from PIL import Image
import glob

# This can combines multiple screenshots into a large frame-by-frame composite image. 
# Not sure if I like it yet.

# --- Settings ---
input_files = sorted(glob.glob("Artificial-Data-Test/PressureDriven-VT-2x/rhat-from-closest-point/*.jpeg"))  # order = time sequence
step = 5         # take every Nth file, set to 1 to keep all
max_files = 64  # set to an integer to use only the first N files, or None for all
input_files = input_files[::step]
if max_files is not None:
    input_files = input_files[:max_files]
cols = 4          # number of columns in the grid
gap = 10          # pixel gap between images and around the border
thumb_w = 500     # width each image gets resized to (height auto-scales to match aspect ratio)
bg_color = (255, 255, 255)
output_path = "timeseries_grid.png"

# --- Load and resize ---
imgs = [Image.open(f) for f in input_files]
thumb_h = int(thumb_w * imgs[0].height / imgs[0].width)
imgs_resized = [im.resize((thumb_w, thumb_h)) for im in imgs]

# --- Compute grid size ---
rows = -(-len(imgs) // cols)  # ceil division
grid_w = cols * thumb_w + (cols + 1) * gap
grid_h = rows * thumb_h + (rows + 1) * gap

# --- Paste each image into place ---
canvas = Image.new("RGB", (grid_w, grid_h), bg_color)
for idx, im in enumerate(imgs_resized):
    r, c = divmod(idx, cols)
    x = gap + c * (thumb_w + gap)
    y = gap + r * (thumb_h + gap)
    canvas.paste(im, (x, y))

canvas.save(output_path)