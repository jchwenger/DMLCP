from py5canvas import *

ALIGN_OPTIONS = [CENTER, LEFT, RIGHT]
VALIGN_OPTIONS = [CENTER, TOP, BOTTOM]


def parameters():
    # Note: all these will be turned to lowercase when working on the object, so
    # we only use caps here so that the GUI looks nicer
    return {
        "Show": True,
        "Le text": ("[]", {"multiline": True, "buf_length": 2024}),
        "Size": (400, {"min": 50, "max": 500}),
        "X": (0.0, {"min": -100, "max": 100}),
        "Y": (0.0, {"min": -100, "max": 100}),
        "Color": ([255, 0, 0], {"type": "color"}),
        # this will create a select menu, yielding the **index** (0, 1,...)!
        "Align": (
            0,
            {
                "selection": ALIGN_OPTIONS,
            },
        ),
        "Vertical Align": (
            0,
            {
                "selection": VALIGN_OPTIONS,
            },
        ),
        # it is possible to nest things
        "Other": {
            "Helper": False,
            "Helper radius": 5.0,
            "Background": (100, {"min": 0, "max": 255}),
        },
    }


def setup():
    # setting this to `False` will stretch the canvas when going fullscreen
    sketch.keep_aspect_ratio = False
    create_canvas(256, 512, 300)


def draw():
    # once processed and by default, parameters are accessed with the labels
    # converted to lowercase and spaces replaced by underscores.

    background(params.other.background)

    # translate to the centre
    translate(width / 2, height / 2)

    text_size(params.size)
    fill(params.color)
    text_align(ALIGN_OPTIONS[params.align], VALIGN_OPTIONS[params.vertical_align])

    if params.show:
        # draw a dot
        if params.other.helper:
            push()
            fill(255)
            no_stroke()
            circle(params.x, params.y, params.other.helper_radius)
            no_fill()
            stroke(0, 255, 0)
            b = text_bounds(params.le_text)
            # print(b)
            rect(b.pos, b.size)
            pop()

        text(
            params.le_text,
            params.x + (sin(frame_count / 20) * 20),
            params.y,
        )


def key_pressed(key, modifier):
    if key == " ":
        # SPACE toggles fullscreen
        # (hiding the gui when fullscreen
        toggle_fullscreen(toggle_gui=True)


run()
