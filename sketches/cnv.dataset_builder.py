from py5canvas import *
import pathlib

# 0: rectangles, 1: ellipses, 2: triangles
sample_class = 0

# the number of images to record
n_images = 10

# bw or colour?
bnw = False

out_dir = pathlib.Path(f"datasets/synthetic/{sample_class}")
out_dir.mkdir(exist_ok=True, parents=True)


def setup():
    create_canvas(512, 512)
    frame_rate(60)
    background(255)


def draw():
    background(255)

    if bnw:
        fill(random(255), random(255))
    else:
        fill(random(255), random(255), random(255), random(255))

    rotation = random(0, 2 * PI)
    quarter_width = width / 2
    half_height = height / 2

    # draw
    if sample_class == 0:
        rect_mode(CENTER)
        translate(center)
        rotate(rotation)
        rect(
            random(-quarter_width, quarter_width),
            random(-half_height, half_height),
            random(10, quarter_width),
            random(10, half_height),
        )
    elif sample_class == 1:
        translate(center)
        rotate(rotation)
        ellipse(
            random(-quarter_width, quarter_width),
            random(-half_height, half_height),
            random(10, quarter_width),
            random(10, half_height),
        )
    else:
        centre = create_vector(random(0, width), random(0, height))
        size = random(10, width / 2)
        draw_equilateral_triangle(centre, size, rotation)

    # save image
    fname = out_dir / f"{sample_class}_{frame_count:03}.png"
    print(f"{fname}")
    save_image(fname)

    # stop the
    if frame_count >= n_images:
        exit()


def draw_equilateral_triangle(centre, size, rotation):
    begin_shape()
    for i in range(3):
        angle = rotation + TWO_PI / 3 * i
        x = centre[0] + size / 2 * cos(angle)
        y = centre[1] + size / 2 * sin(angle)
        vertex(x, y)
    end_shape(close=True)


run()

# IDEAS, for improvement:
#  - Obviously, making better/more interesting shapes could be nice.
#  - Automating the process: instead of having to run the code manually for
#    every class, why not modify the code so that it automatically produces
#    `n_images` for each class, then stops.
#  - One can of course expand the number of classes, creating other
#    shapes/patterns. In this case, it might be best to encapsulate the drawing
#    into functions (`def draw_rectangles`, `def draw_ellipses`).
#  - Another good thing to do: instead of using the index for the folder name,
#    create a list with names (["rectangle", "ellipse", ...]): that can then be
#    picked up automatically by PyTorch as a label!
#  - One could also imagine adding yet another subdirectory using the timestamp
#    of the run, instead of overwriting things every time?
