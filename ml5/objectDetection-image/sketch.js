/*
  Data and machine learning for creative practice (DMLCP)

  Object detection on a static image

  (Deprecated: this is not using ml5js v1!)
  Original sketch: https://editor.p5js.org/ml5/sketches/ObjectDetector_COCOSSD_single_image
  Reference here: https://archive-docs.ml5js.org/#/reference/object-detector
  And other examples: https://archive-docs.ml5js.org/#/reference/object-detector?id=examples
*/

let detector,
    pic,
    detections = [];

// Load our model & image before setup
function preload() {
  // Load our object dection model `cocossd` – you could use `yolo`
  detector = ml5.objectDetector('cocossd');
  pic = loadImage('images/dog.jpg');

  // Load our image, try the other ones in the directory!
  // IDEA: import, or even create, your own images, see what the model is able
  //       to recognise?
  // IDEA: how about implementing more of an interface to load images
  //       dynamically? This could be, for instance, loading all the images in
  //       the `images/` folder, and allowing the user to switch between
  //       images. Or even use an API to pull images from the Internet?
  // IDEA: how about instantiating *two* networks, `cocossd` and `yolo`, and
  //       display both predictions? Ideally you'd want to make clear to the
  //       user which model predicted which bounding box?

}

function setup() {
  createCanvas(pic.width, pic.height);

  // IDEA: here, instead of using an existing image, you could *create one*
  //       using createGraphics, and see what the network recognises as an
  //       image or not? Try with just random colours everywhere, for instance,
  //       how does the network react?
  // IDEA: more advanced, you could imagine an interface where by pressing a
  //       key some randomness, or various shapes, are *added* to an existing
  //       image (or even something where the user can draw on the image) –
  //       then, the network would be requested to classify the new image (you
  //       probably need to import the image into a graphics object you can
  //       draw onto, or extract the canvas itself into an image object, that
  //       can be passed to the network). The interesting thing is that this
  //       would allow people to see how the new patterns gradually degrade the
  //       network's performance!

  console.log("Setting up, about to detect...");
  // start our detector, give the callback function of gotResults()
  detector.detect(pic, gotResults);
}

function draw() {

  background(0);

  // Draw dog photo
  image(pic, 0, 0);

  // Loop through all our detections and draw them
  for (let object of detections) {

    // We use lerp to color the border somewhere between red and green based on
    // the confidence of the prediction
    stroke(lerpColor(color(255,0,0), color(0, 255, 0), object.confidence));
    strokeWeight(3);
    noFill();

    // Draw rectangle around a detection
    rect(object.x, object.y, object.width, object.height);

    // We could also use the normalised values, these represent the percentage
    // across the screen as a value between 0 and 1 – so we multiply these out
    // by the width and height of the canvas:
    // rect(object.normalized.x * pic.width,
    //      object.normalized.y * pic.height,
    //      object.normalized.width * pic.width,
    //      object.normalized.height * pic.height);

    // Draw the label
    fill(255);
    noStroke();
    textSize(24);
    text(object.label, object.x + 10, object.y + 24);
  }

  // IDEA: could you think of a way to show the user (without requiring them to
  //       look into the console) that the model is currently processing the image?
  //       (You could use the fact that `detections` is empty/has a length of 0).
  // IDEA: there are some images for which the network does not detect anything!
  //       You could also think of a way to handle that case? (Here, you might need
  //       to use the `gotResults` callback to flip a variable, for instance: ok,
  //       detections is empty *and yet* gotResults has been called, so that means
  //       our network sees nothing!
  // IDEA: instead of only using the model as intended (image detection), how about
  //       using, say, only the labels, or only the bounding boxes, as material for
  //       some generative art where we would not necessarily see the original photo?
}

// Our callback function!
function gotResults(err, results) {
  if (err) {
    console.log("We had an error with the detection:", err);
    return;
  }
  // Remember our detections so that we can draw them in draw()
  detections = results;
  console.log("Huzzah! It detected!");
  console.log(detections); // if you want to see what `detections` look like
}
