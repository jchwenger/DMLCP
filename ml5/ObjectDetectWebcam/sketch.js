/*
  Data and machine learning for creative practice (DMLCP)

  Object detection with a Webcam

  (Deprecated: this is not using ml5js v1!)
  Original sketch: https://editor.p5js.org/ml5/sketches/ObjectDetector_COCOSSD_Video
  Reference here: https://archive-docs.ml5js.org/#/reference/object-detector
  And other examples: https://archive-docs.ml5js.org/#/reference/object-detector?id=examples
*/

let video,
    detector,
    detections = [];

function preload() {
  // Load our object dection model `cocossd` – you could use 'yolo'
  detector = ml5.objectDetector('cocossd');
}

function setup() {
  createCanvas(640, 480);

  // Load our webcam feed, when the video is ready the videoReady callback will
  // be called
  video = createCapture(VIDEO, videoReady);
  video.size(640, 480);
  video.hide();
}

function videoReady(stream) {
  // We now know the video is ready, so we'll start the detection.
  // Start our detector, pass in the callback function 'gotResults' called once
  // the detection is done
  detector.detect(video, gotResults);
}

function draw() {
  background(0);

  // Draw webcam frame
  image(video, 0, 0);

  // IDEA: instead of only using the model as intended (image detection), how
  //       about using, say, only the labels, or only the bounding boxes, as
  //       material for some generative art where we would not necessarily even
  //       see the video feed?
  // IDEA: note that the labels could be used poetically: for instance, you
  //       could choose a large text file (containing one or more novels,
  //       poetry, other kinds  of text), load it in memory, and then *search*
  //       through it for a sentence con- taining the label of the object being
  //       detected?
  // IDEA: something that I've seen implemented somewhere, is the idea of using
  //       an ex- ternal API (an image search engine) to look for images of
  //       what is being de- tected (person, apple, etc.), pull that from the
  //       internet, and display the random image on top of the video feed in
  //       the bounding box of the detected object?

  // Loop through all our detections
  for (let object of detections) {

    // We use lerp to color the border somewhere between red and green based on
    // the confidence of the prediction
    stroke(lerpColor(color(255,0,0), color(0, 255, 0), object.confidence));
    strokeWeight(3);
    noFill();

    // Draw a rectangle around the recognized object
    rect(object.x, object.y, object.width, object.height);

    // In this commented version, we use the normalised values, these represent
    // the percentage across the screen as a value between 0 and 1 – so we
    // multiply these out by the width and height of the canvas. this will be
    // useful in case we want to rescale the video:
    // rect(object.normalized.x * video.width,
    //      object.normalized.y * video.height,
    //      object.normalized.width * video.width,
    //      object.normalized.height * video.height);

    // Draw the label
    fill(255);
    noStroke();
    textSize(24);
    text(object.label, object.x + 10, object.y + 24);
  }
}

// we can use mouseClicked to
function mouseClicked() {
  if (detector && video) {
    // the model returns a Promise, see:
    // https://www.youtube.com/playlist?list=PLRqwX-V7Uu6bKLPQvPRNNE65kBL62mVfx
    detector.detect(video)
      .then((res) => {
        console.log(res); // one detection
      });
  } else {
    console.log("Model or video not ready yet...");
  }
}

function gotResults(err, results) {
  if (err) {
    console.log("We had an error with the detection:", err);
  }

  // Remember our detections so that we can draw them in draw()
  detections = results;

  // Recursion! <3 By default, no new detection will be done so we need to
  // restart the detection again (as we did in videoReady(...)) to keep
  // detecting each frame:
  detector.detect(video, gotResults);
}

// IDEA: (advanced) One could imagine modifying this sketch to use a 'friend'
//       network to counteract the other 'external' (or 'enemy') network, in
//       order to prevent the 'enemy' network from detecting a person? One way
//       of going about this could be the following:
//       - instantiate *two*, rather than just one network;
//       - use the 'friend' network to do detection on the video feed
//       - if the 'friend' network detects one or more 'person' in the feed
//         (you need to study the labels, and the structure of the 'results'
//         array to do that), modify the current video feed by, for instance,
//         drawing a full box where the person is (using the detection's
//         x/y/width/height), or some other strategy to change that area of the
//         image)
//       - then instead of feeding the normal video to the 'enemy' network,
//         feed that modified feed, and see if you are able to prevent the
//         'enemy' network from seeing anybody?
//       It could be nice to have two videos next to each other, one where one
//       sees the 'friend' network detecting the person, and the other with the
//       modified video feed and the 'enemy' network no longer detecting
//       anybody.
