<a id="canvas"></a>

# canvas

Simplistic utilty to mimic [P5js](https://p5js.org) in Python/Jupyter notebooks.

## Dependencies
This module depends on matplotlib, numpy and and pycairo.
To install these with conda do:
```
conda install -c conda-forge matplotlib
conda install -c conda-forge pycairo
```

When using Google Colab, matplotlib and numpy will already be installed. To
install pycairo add and execute the following in a code cell:
```
!apt-get install libcairo2-dev libjpeg-dev libgif-dev
!pip install pycairo
```

## Usage
Place the `canvas.py` file in the same directory as your notebook.
If using Google Colab fetch the latest version of the module with
```
!wget https://raw.githubusercontent.com/colormotor/DMLAP/main/python/canvas.py
```

## Example
The following is the conversion of a simple example in P5js to the `canvas` API.
In Javascript we may have:

```Javascript
function setup() {
  createCanvas(512, 512);
  // Clear background to black
  background(0);
  // Set stroke only and draw circle
  stroke(128);
  noFill();
  strokeWeight(5);
  circle(width/2, height/2, 200);
  // Draw red text
  fill(255, 0, 0);
  noStroke();
  textSize(30);
  textAlign(CENTER);
  text("Hello world", width/2, 40);
}

function draw() {
}
```

The equivalent Python version will be:

```Python
import canvas

# Create our canvas object
c = canvas.Canvas(512, 512)

# Clear background to black
c.background(0)
# Set stroke only and draw circle
c.stroke(128)
c.no_fill()
c.stroke_weight(5)
c.circle(c.width/2, c.height/2, 100)
# Draw red text
c.fill(255, 0, 0)
c.text_size(30)
c.text([c.width/2, 40], "Hello world", center=True)
c.show()
```

<a id="canvas.Canvas"></a>

## Canvas Objects

```python
class Canvas()
```

Creates a a pycairo surface that behaves similarly to p5js

<a id="canvas.Canvas.__init__"></a>

#### \_\_init\_\_

```python
def __init__(width, height)
```

Initialize Canvas with given `width` and `height`

<a id="canvas.Canvas.set_color_scale"></a>

#### set\_color\_scale

```python
def set_color_scale(scale)
```

Set color scale, e.g. if we want to specify colors in the `0`-`255` range, scale would be `255`,
or if the colors are in the `0`-`1` range, scale will be `1`

<a id="canvas.Canvas.stroke_weight"></a>

#### stroke\_weight

```python
def stroke_weight(w)
```

Set the line width

<a id="canvas.Canvas.line_cap"></a>

#### line\_cap

```python
def line_cap(cap)
```

Specify the 'cap' for lines.

**Arguments**:

- `cap` _string_ - can be one of "butt", "round" or "square"

<a id="canvas.Canvas.translate"></a>

#### translate

```python
def translate(*args)
```

Translate by specifying `x` and `y` offset.

**Arguments**:

  The offset can be specified as an array/list (e.g `c.translate([x,y])`
  or as single arguments (e.g. `c.translate(x, y)`)

**Returns**:

  Nothing

<a id="canvas.Canvas.scale"></a>

#### scale

```python
def scale(*args)
```

Apply a scaling transformation.

**Arguments**:

  Providing a single number will apply a uniform transformation.
  Providing a pair of number will scale in the x and y directions.
  The scale can be specified as an array/list (e.g `c.scale([x,y])`
  or as single arguments (e.g. `c.scale(x, y)`)'''
  

**Returns**:

- `type` - nothing

<a id="canvas.Canvas.rotate"></a>

#### rotate

```python
def rotate(theta)
```

Rotate by `theta` radians

<a id="canvas.Canvas.rotate_deg"></a>

#### rotate\_deg

```python
def rotate_deg(deg)
```

Rotate using degrees

<a id="canvas.Canvas.rectangle"></a>

#### rectangle

```python
def rectangle(*args)
```

Draw a rectangle given top-left corner, width and heght.

**Arguments**:

  Input arguments can be in the following formats:
  - `[topleft_x, topleft_y], [width, height]`,
  - `[topleft_x, topleft_y], width, height`,
  - `topleft_x, topleft_y, width, height`

<a id="canvas.Canvas.rect"></a>

#### rect

```python
def rect(*args)
```

Draw a rectangle given top-left corner, width and heght.

**Arguments**:

  Input arguments can be in the following formats:
  - `[topleft_x, topleft_y], [width, height]`,
  - `[topleft_x, topleft_y], width, height`,
  - `topleft_x, topleft_y, width, height`

<a id="canvas.Canvas.quad"></a>

#### quad

```python
def quad(*args)
```

Draws a quadrangle given four points

**Arguments**:

  Input arguments can be in the following formats:
  - `a, b, c, d` (Four points specified as lists/tuples/numpy arrays
  - `x1, y1, x2, y2, x3, y3, x4, y4`

<a id="canvas.Canvas.triangle"></a>

#### triangle

```python
def triangle(*args)
```

Draws a triangle given three points

**Arguments**:

  Input arguments can be in the following formats:
  - `a, b, c` (Four points specified as lists/tuples/numpy arrays
  - `x1, y1, x2, y2, x3, y3`

<a id="canvas.Canvas.circle"></a>

#### circle

```python
def circle(*args)
```

Draw a circle given center and radius

**Arguments**:

  Input arguments can be in the following formats:
  - `[center_x, center_y], radius`,
  - `center_x, center_y, raidus`

<a id="canvas.Canvas.ellipse"></a>

#### ellipse

```python
def ellipse(*args)
```

Draw an ellipse with center, width and height.

**Arguments**:

  Input arguments can be in the following formats:
  - `[center_x, center_y], [width, height]`,
  - `[center_x, center_y], width, height`,
  - `center_x, center_y, width, height`

<a id="canvas.Canvas.arc"></a>

#### arc

```python
def arc(*args)
```

Draw an arc given the center of the ellipse `x, y`
the size of the ellipse `w, h` and the initial and final angles
in radians  `start, stop`.

**Arguments**:

  Input arguments can be in the following formats:
  -`x, y, w, h, start, stop`
  -`[x, y]', '[w, h]', '[start, stop]'
  -`[x, y]', w, h, start, stop`

<a id="canvas.Canvas.line"></a>

#### line

```python
def line(*args)
```

Draw a line between given its end points.

**Arguments**:

  Input arguments can be in the following formats:
  `[x1, y1], [x2, y2]`,
  `x1, y1, x2, y2`

<a id="canvas.Canvas.begin_shape"></a>

#### begin\_shape

```python
def begin_shape()
```

Begin drawing a compound shape

<a id="canvas.Canvas.end_shape"></a>

#### end\_shape

```python
def end_shape()
```

End drawing a compound shape

<a id="canvas.Canvas.load_image"></a>

#### load\_image

```python
def load_image(path)
```

Load an image from disk. Currently only supports png! Use external
loading into NumPy instead

<a id="canvas.Canvas.image"></a>

#### image

```python
def image(img, *args, opacity=1.0)
```

Draw an image at position with (optional) size and (optional) opacity

**Arguments**:

- `img` - The input image. Can be either a numpy array or a pyCairo surface (e.g. another canvas).
- `*args` - position and size can be specified with the following formats:
  `x, y`:  position only
  `x, y, w, h`: position and size
  `[x, y]`: position only (also a numpy array or tuple are valid)
  `[x, y], [w, h]`: position and size
  if the position is not specified, the original image dimensions will be used
  
- ``opacity`` - a value between 0 and 1 specifying image opacity.

<a id="canvas.Canvas.shape"></a>

#### shape

```python
def shape(poly_list, closed=False)
```

Draw a shape represented as a list of polylines, see the ~polyline~
method for the format of each polyline

<a id="canvas.Canvas.text"></a>

#### text

```python
def text(pos, text, center=False)
```

Draw text at a given position

**Arguments**:

  if center=True the text will be horizontally centered

<a id="canvas.Canvas.polygon"></a>

#### polygon

```python
def polygon(*args)
```

Draw a *closed* polygon
The polyline is specified as either:
- a list of `[x,y]` pairs (e.g. `[[0, 100], [200, 100], [200, 200]]`)
- a numpy array with shape `(n, 2)`, representing `n` points (a point for each row and a coordinate for each column)

<a id="canvas.Canvas.polyline"></a>

#### polyline

```python
def polyline(*args, closed=False)
```

Draw a polyline.
The polyline is specified as either:
- a list of `[x,y]` pairs (e.g. `[[0, 100], [200, 100], [200, 200]]`)
- a numpy array with shape `(n, 2)`, representing `n` points (a point for each row and a coordinate for each column)

To close the polyline set the named closed argument to `True`, e.g. `c.polyline(points, closed=True)`.

<a id="canvas.Canvas.background"></a>

#### background

```python
def background(*args)
```

Clear the canvas with a given color

<a id="canvas.Canvas.get_image"></a>

#### get\_image

```python
def get_image()
```

Get canvas image as a numpy array

<a id="canvas.Canvas.get_image_grayscale"></a>

#### get\_image\_grayscale

```python
def get_image_grayscale()
```

Returns the canvas image as a grayscale numpy array (in 0-1 range)

<a id="canvas.Canvas.save_image"></a>

#### save\_image

```python
def save_image(path)
```

Save the canvas to an image

<a id="canvas.radians"></a>

#### radians

```python
def radians(x)
```

Get radians given x degrees

<a id="canvas.degrees"></a>

#### degrees

```python
def degrees(x)
```

Get degrees given x radians

<a id="canvas.numpy_to_surface"></a>

#### numpy\_to\_surface

```python
def numpy_to_surface(arr)
```

Convert numpy array to a pycairo surface

<a id="canvas.show_image"></a>

#### show\_image

```python
def show_image(im, size=None, title='', cmap='gray')
```

Display a (numpy) image

<a id="canvas.show_images"></a>

#### show\_images

```python
def show_images(images, ncols, size=None, title='', cmap='gray')
```

Display multiple images in a grid

