/*
  Data and machine learning for creative practice (DMLCP)

  Facial detection on a webcam

  Adapted from Joe McAlister's version

  (Deprecated: this is not using ml5js v1!)
  Original sketch: https://editor.p5js.org/ml5/sketches/FaceApi_Video_Landmarks
  Reference here: https://archive-docs.ml5js.org/#/reference/face-api
  And other examples: https://archive-docs.ml5js.org/#/reference/face-api?id=examples
*/

let faceapi,
    video,
    detections,
    hasLogged = false;
    // to inspect our object just once, see below

// These are our options for detecting faces, provided by ml5.js
const detection_options = {
  withLandmarks: true,
  withDescriptors: false,
}

function setup() {
  createCanvas(600, 338);
  // I use these to downsize a 720p stream, but you can adjust for your webcam
  // if it skews it.

  // Ask for webcam access – with the `webcamReady` callback for when we have
  // access
  video = createCapture(VIDEO, webcamReady);
  video.size(width, height); // Set size to be equal to canvas
  video.hide();              // Hide DOM element
}

function webcamReady(stream) {
  // Load the faceapi model – with a `modelReady` callback
  faceapi = ml5.faceApi(video, detection_options, modelReady)
}

function draw() {
  background(0);

  // Draw picture
  image(video, 0,0, width, height)

  // If we have detections, draw them on the image
  // When we call detect, we are looking for potentially multiple faces, so
  // ml5.js returns an array of objects, therefore here we use a for loop to
  // get each 'person'. Note the for...of loop:
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/for...of
  if (detections) {
    for (let person of detections) {
      drawBox(person);
      drawLandmarks(person);
    }
  }

  // IDEA: like in previous sketches, there is no obligation to display the
  //       video, and you could for instance imagine a blank canvas where
  //       points from the face, or the box itself, can be used in artistic
  //       ways! To draw or influence shapes on the canvas in real time!

}

// Callback for when ml5.js has loaded the model
function modelReady() {
  console.log("Model is ready...");

  faceapi.detect(gotResults);

  // Ask ml5 to detect a faces in the video stream previously provided, then
  // call the `gotResults` callback
                              
}

// ml5.js has determined if there's a face
function gotResults(err, result) {

  // Check if ml5.js returned an error – if so print to console and stop
  if (err) {
    console.log(err)
    return
  }

  // If it gets here we are okay, so store results in the detections variable.
  detections = result;

  // Only log once!
  if (!hasLogged && detections.length > 0) {
    // This is an object – have a look at it in the console!
    console.log("The predictions object:");
    console.log(result);
    hasLogged = true;
  }

  // Recursion! <3 We call face detect again
  faceapi.detect(gotResults);
}


// Draw our elements on the image, a box and face feature locations
function drawBox(detections){
  push(); // Precaution: make sure these changes stay in this function

  noFill();
  strokeWeight(2);
  stroke(161, 95, 251);

  const alignedRect = detections.alignedRect;
  const {_x, _y, _width, _height} = alignedRect._box; // The _variables are the actual properties of the `_box` object!
  rect(_x, _y, _width, _height);

  pop();
}

function drawLandmarks(detections){
  /*
    In this example we use forEach(), this is a quick way of looping through
    the objects in an array. It will in this case return each value in the
    array sequentially as the local variable "item".

    In this case, it's equivalent of doing:
      for (let i = 0; i < detections.parts.mouth.length; i++) {
        let item = detections.parts.mouth[i];
        vertex(item._x, item._y);
      }

    or:

      for (let item of detections.parts.mouth) {
        //...
      }
  */

  push(); // Keep the styles within this function!

  noFill();
  strokeWeight(2);
  stroke(161, 95, 251);

  // mouth
  beginShape();
  detections.parts.mouth.forEach(item => {
    vertex(item._x, item._y);
  })
  endShape(CLOSE);

  // nose
  beginShape();
  detections.parts.nose.forEach(item => {
    vertex(item._x, item._y);
  })
  endShape(CLOSE);

  // left eye
  beginShape();
  detections.parts.leftEye.forEach(item => {
    vertex(item._x, item._y);
  })
  endShape(CLOSE);

  // right eye
  beginShape();
  detections.parts.rightEye.forEach(item => {
    vertex(item._x, item._y);
  })
  endShape(CLOSE);

  // right eyebrow
  beginShape();
  detections.parts.rightEyeBrow.forEach(item => {
    vertex(item._x, item._y);
  })
  endShape();

  // left eye
  beginShape();
  detections.parts.leftEyeBrow.forEach(item => {
    vertex(item._x, item._y);
  })
  endShape();

  pop();

}
