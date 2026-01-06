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
    hasLogged = false, // Used to log preds only once
    options = { maxFaces: 1, refineLandmarks: false, flipped: false };
    // note that you must set the limit of detectable faces in advance!
    // landmarks refinement: more precision but slower (more compute!)
    // to flip, also flip the video below

function preload() {
  faceMesh = ml5.faceMesh(options);
}

function setup() {
  createCanvas(640, 480);
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

  drawKeypoints();

  // We call our function to draw all keypoints
  // IDEA: like in previous sketches, there is no obligation to display the
  //       video, and you could for instance imagine a blank canvas where a few
  //       points from the face are used to draw vanishing circles, using the
  //       same logic as when you want a circle to leave a trail behind it when
  //       it moves?

}

function gotFaces(results) {
  faces = results;

  // Log `results` to see its contents
  if (!hasLogged && results.length > 0) {
    console.log("The predictions object:");
    console.log(results);
    hasLogged = true;
  }

}


// A function to draw ellipses over the detected keypoints
function drawKeypoints() {

  if (faces.length > 0) {

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
