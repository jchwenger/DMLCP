from py5canvas import *


# IMPORTANT PYTHON NOTES:
# - The code inside `setup` runs only once. The code in `draw` runs again and
#   again forever (here approximately 60 times per second).
# - Indentation (leading spaces/tabs) matters! All the code
#   inside `setup` or `draw` MUST be indented with one tab (4 spaces).   
# - When you select your sketch, cmd/ctrl+q will stop it ('quit')

def setup():
    # create a canvas, width: 512, height: 512
    # (try changing those values!)
    create_canvas(512, 512)
    frame_rate(60)

    # this paints the background in black **once** (if you comment this, you'll
    # see that the sketch starts by being grey, before moving to black – that's
    # because the default background is grey/empty...)
    # (the single number here can go from 0, black, to 255, white)
    background(0)


def draw():
    # the four numbers determines the colour (Red, Green, Blue, Alpha):
    # https://rgbcolorpicker.com/ redrawing the background with alpha will
    # create the "trail effect" (we 'gradually' add black on top of what we
    # draw below, things that we draw in the past will have more layers of
    # semi-transparent background on top of them – they will slowly fade
    # (try removing the 8 (and the ',' before it) to see the difference)
    background(0, 0, 0, 8)

    # center of screen: `width/height` contain the numbers we define in
    # `create_canvas`, and the coordinates count from the **top left corner**
    # (right for x/width, down for y/height)
    translate(width / 2, height / 2)

    # rotating the canvas
    # (try changing the number, or adding `-` before)
    rotate(frame_count * 0.05)

    # draw a circle
    # `fill` takes 1, 3 or 4 arguments, like background
    fill(255, 0, 0)

    # `stroke` draws an outline around objects, same colour system as `fill`
    # (to change the thickness, use `stroke_weight(2)` – the number of pixels
    stroke(255)

    # circle takes three arguments: x, y, radius
    circle(0, 100, 80)

    # Something to try: uncomment the next lines by removing the leading '# '
    # (select them & cmd/ctrl + / )

    # text_align(CENTER)
    # text_size(200)
    # no_stroke()
    # fill(255)
    # text("?", 0, 0)


run()

# IDEAS, to make it your own:
# - This week, we are not drilling into this particular sketch, but if you feel
#   so inclined, feel free to look at the py5canvas documentation before next
#   week: https://github.com/colormotor/py5canvas/tree/main/docs, it is quite
#   similar to p5.js/Processing!
# - If you are coming back to this later, I recommend trying:
#   - adding other shapes, for instance:
#     `triangle(x1, y1, x2, y2, x3, y3)`: https://github.com/colormotor/py5canvas/tree/main/docs#triangle
#     `ellipse(center_x, center_y, width, height)`: https://github.com/colormotor/py5canvas/tree/main/docs#ellipse
#     `line(x1, y1, x2, y2)`: https://github.com/colormotor/py5canvas/tree/main/docs#line
#     `square(x, y, size)`: https://github.com/colormotor/py5canvas/tree/main/docs#square
#     `rect(topleft_x, topleft_y, width, height)`: https://github.com/colormotor/py5canvas/tree/main/docs#rect
#   - playing with `translate` and `rotate`. Note that these commands are
#     additive: if you `translate` once by `x1, y1`, then again by `x2, y2`,
#     you will have translated in total by `x1+x2, y1+y2`. To avoid this, so to
#     'cancel' the effect of translate and rotate, you need to 'wrap' your
#     commands with `push` and `pop` like this:
#       # like opening a parenthesis
#       push()
#       translate(100, 50)
#       # in here we have translated, this circle will be drawn at 100, 50
#       circle(0, 0, 20)
#       # like closing a parenthesis
#       pop()
#       # here we are back to before `push`
