/*
  Data and machine learning for creative practice (DMLCP)

  Face mesh with a webcam

  Fun experiments with facemesh and particle systems:
  (still using v0, but that shouldn't matter too much)
  https://www.youtube.com/live/931bqqpQqvI?si=YRJL9SVDJAbgcdtp&t=7379

  Reference here: https://docs.ml5js.org/#/reference/facemesh
*/


let video,
    faceMesh,
    faces = [],
    options = { maxFaces: 1, refineLandmarks: false, flipped: false },
    mode = 0; // drawing mode, loop through by pressing the space bar
    // note that you must set the limit of detectable faces in advance!
    // landmarks refinement: more precision but slower (more compute!)
    // to flip, also flip the video below

async function setup() {
  createCanvas(640, 480);
   
  faceMesh = await ml5.faceMesh(options);

  video = createCapture(VIDEO); // to flip, add: { flipped: true }
  video.size(width, height);

  // Start detecting faces from the webcam video
  faceMesh.detectStart(video, gotFaces);

  // This sets up an event that fills the global variable "faces"
  // with an array every time new predictions are made
  // Hide the video element, and just show the canvas
  video.hide();
}

function draw() {
  image(video, 0, 0, width, height);

  // Depending on the mode, we use the keypoints in various ways
  switch(mode) {
    case 0:
      drawKeypoints();
      break;
    case 1:
      drawKeypointsFromParts();
      break;
    case 2:
      drawBoundingBoxes();
      break;
    case 3:
      drawBoundingBoxesFromParts();
      break;
    case 4:
      drawShapesFromParts();
      break;
  }

  // We call our function to draw all keypoints
  // IDEA: like in previous sketches, there is no obligation to display the
  //       video, and you could for instance imagine a blank canvas where a few
  //       points from the face are used to draw vanishing circles, using the
  //       same logic as when you want a circle to leave a trail behind it when
  //       it moves?

}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  push();
  for (let i = 0; i < faces.length; i++) {
    const keypoints = faces[i].keypoints;

    // IDEA: in v0, you could get an 'unscaled' mesh (unnormalised version)
    //       that would then produce a smaller, fixed version of the face
    //       (e.g. in the upper left corner of the sketch), as the values
    //       were restricted to be always in the same range. You could still
    //       achieve that by using the width and the height of the sketch to
    //       normalise the values of the landmarks yourself! This in turn
    //       could be used if you wanted a face mesh that moves like the
    //       person being filmed, but that stays fixed (instead of being
    //       superimposed to the same location in the image). Proprely scaled
    //       again, this 'static' yet moving face could occupy the whole
    //       canvas, like a mirror!

    // Draw facial keypoints.
    for (let j = 0; j < keypoints.length; j++) {
      // the coordinates are given in 3D! Here we only use x & y
      const {x, y, z} = keypoints[j];

      fill(0, 255, 0);
      ellipse(x, y, 5, 5);
    }

  }
  pop();
}

// A function to draw bounding boxes around detected faces
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/faceMesh-bounding-box-p5-2.0/sketch.js
function drawBoundingBoxes() {
  push();
  // Draw the faces' bounding boxes
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];
    let x = face.box.xMin;
    let y = face.box.yMin;
    let w = face.box.width;
    let h = face.box.height;
    let centerX = (face.box.xMin + face.box.xMax) / 2; // average of xMin and xMax
    let centerY = (face.box.yMin + face.box.yMax) / 2; // average of yMin and yMax

    stroke(0, 255, 0);
    fill(0, 255, 0, 50);
    rect(x, y, w, h);
    text(i, x, y - 10);

    // Draw the center of the face
    noStroke();
    fill(255, 0, 0);
    circle(centerX, centerY, 10);
  }
  pop();
}

// A function to draw keypoints on parts of faces
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/faceMesh-keypoints-from-parts-p5-2.0/sketch.js
function drawKeypointsFromParts() {
  push();
  // draw the faces' bounding boxes
  for (let j = 0; j < faces.length; j++) {
    let face = faces[j];

    strokeWeight(5);
    // draw the lips
    stroke(255, 0, 255);
    for (let i = 0; i < face.lips.keypoints.length; i++) {
      let keypoint = face.lips.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      point(x, y);
    }
    // draw the left eye
    stroke(255, 255, 0);
    for (let i = 0; i < face.leftEye.keypoints.length; i++) {
      let keypoint = face.leftEye.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      point(x, y);
    }
    // draw the left eyebrow
    stroke(0, 255, 0);
    for (let i = 0; i < face.leftEyebrow.keypoints.length; i++) {
      let keypoint = face.leftEyebrow.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      point(x, y);
    }
    // draw the right eye
    stroke(0, 255, 255);
    for (let i = 0; i < face.rightEye.keypoints.length; i++) {
      let keypoint = face.rightEye.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      point(x, y);
    }
    // draw the right eyebrow
    stroke(0, 0, 255);
    for (let i = 0; i < face.rightEyebrow.keypoints.length; i++) {
      let keypoint = face.rightEyebrow.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      point(x, y);
    }
    // draw the face oval
    stroke(255, 0, 0);
    for (let i = 0; i < face.faceOval.keypoints.length; i++) {
      let keypoint = face.faceOval.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      point(x, y);
    }
  }

  pop();
}

// A function to draw bounding boxes around parts of faces
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/faceMesh-parts-bounding-box-p5-2.0/sketch.js
function drawBoundingBoxesFromParts() {
  push();

  // draw the faces' bounding boxes
  for (let i = 0; i < faces.length; i++) {
    let face = faces[i];

    // draw the bounding box of face parts
    fill(0, 255, 0, 50);
    stroke(0, 255, 0);
    rect(face.lips.x, face.lips.y, face.lips.width, face.lips.height);
    rect(
      face.leftEye.x,
      face.leftEye.y,
      face.leftEye.width,
      face.leftEye.height
    );
    rect(
      face.leftEyebrow.x,
      face.leftEyebrow.y,
      face.leftEyebrow.width,
      face.leftEyebrow.height
    );
    rect(
      face.rightEye.x,
      face.rightEye.y,
      face.rightEye.width,
      face.rightEye.height
    );
    rect(
      face.rightEyebrow.x,
      face.rightEyebrow.y,
      face.rightEyebrow.width,
      face.rightEyebrow.height
    );
    rect(
      face.faceOval.x,
      face.faceOval.y,
      face.faceOval.width,
      face.faceOval.height
    );

    // draw the center points of face parts
    noStroke();
    fill(255, 0, 0);
    circle(face.lips.centerX, face.lips.centerY, 10);
    circle(face.leftEye.centerX, face.leftEye.centerY, 10);
    circle(face.leftEyebrow.centerX, face.leftEyebrow.centerY, 10);
    circle(face.rightEye.centerX, face.rightEye.centerY, 10);
    circle(face.rightEyebrow.centerX, face.rightEyebrow.centerY, 10);
    circle(face.faceOval.centerX, face.faceOval.centerY, 10);
  }

  pop();
}

// A function to draw shapes around parts of faces
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/faceMesh-shapes-from-parts-p5-2.0/sketch.js
function drawShapesFromParts() {
  push();

  // draw the faces' bounding boxes
  for (let j = 0; j < faces.length; j++) {
    let face = faces[j];

    noFill();

    // draw the lips
    stroke(255, 0, 255);
    beginShape();
    for (let i = 0; i < face.lips.keypoints.length; i++) {
      let keypoint = face.lips.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      vertex(x, y);
    }
    endShape(CLOSE);

    // draw the left eye
    stroke(255, 255, 0);
    beginShape();
    for (let i = 0; i < face.leftEye.keypoints.length; i++) {
      let keypoint = face.leftEye.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      vertex(x, y);
    }
    endShape(CLOSE);

    // draw the left eyebrow
    stroke(0, 255, 0);
    beginShape();
    for (let i = 0; i < face.leftEyebrow.keypoints.length; i++) {
      let keypoint = face.leftEyebrow.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      vertex(x, y);
    }
    endShape(CLOSE);

    // draw the right eye
    stroke(0, 255, 255);
    beginShape();
    for (let i = 0; i < face.rightEye.keypoints.length; i++) {
      let keypoint = face.rightEye.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      vertex(x, y);
    }
    endShape(CLOSE);

    // draw the right eyebrow
    stroke(0, 0, 255);
    beginShape();
    for (let i = 0; i < face.rightEyebrow.keypoints.length; i++) {
      let keypoint = face.rightEyebrow.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      vertex(x, y);
    }
    endShape(CLOSE);

    // draw the face oval

    beginShape();
    for (let i = 0; i < face.faceOval.keypoints.length; i++) {
      let keypoint = face.faceOval.keypoints[i];
      let x = keypoint.x;
      let y = keypoint.y;
      vertex(x, y);

      // display the index
      noStroke();
      fill(255, 0, 0);
      textSize(8);
      text(i, x + 10, y);
    }
    stroke(255, 0, 0);
    noFill();
    endShape(CLOSE);
  }

  pop();
}

function gotFaces(results) {
  faces = results;
}

// A click of the mouse logs the faces
function mousePressed() {
  // Log `faces` to see its contents
  console.log("The predictions object:");
  console.log(faces);
}

function keyPressed() {
  if (key === ' ') {
    mode = (mode + 1) % 5;
    console.log(`mode is now ${mode}`);
  }
}

// IDEA: the predictions object comes with a bounding box, accessible under the
//       `.boundingBox` property. This is also an object, with the x y coordinates of
//       the four corners as arrays: topLeft, topRight, bottomLeft, bottomRight. A
//       nice exercise could be to write a function called drawBoundingBox, similar to
//       drawKeyPoints, that would:
//         - set the rectMode to CORNERS (using a push/pop logic for security)
//         - loop through all the predictions
//         - fetch the topLeft and bottomRight coordinates
//         - and draw the box! (You would call that function in draw after drawKeypoints.)
//       Note that you could imagine doing something different with that, just
//       as you could use the various face points in different ways. It is
//       probably particularly interesting if you focus on only some points
//       (maybe one in each cheek?), or perhaps three-four points that would allow
//       you to define an arc, a spline, or a Bézier curve?
