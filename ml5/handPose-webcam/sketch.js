/*
  Data and machine learning for creative practice (DMLCP)

  Handpose webcam demo

  Originally from here (deprecated): https://editor.p5js.org/ml5/sketches/Handpose_Webcam
  Reference here: https://docs.ml5js.org/#/reference/handpose
  And other examples: https://docs.ml5js.org/#/reference/handpose?id=examples
*/

let video,
    handPose,
    hands = [],
    mode = 0, // switch between modes pressing the space bar
    isDetecting = true; // detection mode: start/stop detection, press 's' to toggle

async function setup() {
  createCanvas(640, 480);
  handPose = await ml5.handPose();
  // Create the webcam video and hide it
  video = createCapture(VIDEO);
  video.hide();
  // Start detecting hands from the webcam video
  handPose.detectStart(video, gotHands);
  // Get the skeletal connection information
  connections = handPose.getConnections();
}

function draw() {

  image(video, 0, 0, width, height);

  // IDEA: like in previous sketches, there is no obligation to display the video,
  //       and you could for instance imagine a blank canvas where points from the
  //       hand, can be used in artistic ways! To draw or influence shapes on
  //       the canvas in real time!
  // IDEA: (advanced) in this vein, it might be possible to train a sound model
  //       on Teachable Machine using different hand poses, and then combine the
  //       local sound sketch with this one, where the landmarks control
  //       animations and sound!

  switch(mode) {
    case 0:
      drawHandKeypoints();
      break;
    case 1:
      drawHandSkeleton();
      break;
    case 2:
      drawPinch();
      break;
  }
}


// A function to draw the keypoints on the detected hands
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/handPose-keypoints-p5-2.0/sketch.js
function drawHandKeypoints() {
  push(); // Precaution: styles remain within this function
  noStroke();
  fill(255,0,0); // Set colour of circle

  // if we have any hand detected, draw it
  if (hands.length > 0) {

    // Draw all the tracked hand points
    for (let i = 0; i < hands.length; i++) {
      let hand = hands[i];

      for (let j = 0; j < hand.keypoints.length; j++) {
        let keypoint = hand.keypoints[j];
        fill(0);
        stroke(0, 255, 0);
        circle(keypoint.x, keypoint.y, 5);
      }

    }

  }

  pop();
}

// A function to draw the hand skeleton
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/handPose-skeletal-connections-p5-2.0/sketch.js
function drawHandSkeleton() {
  push();

  // Draw the skeletal connections
  for (let i = 0; i < hands.length; i++) {
    let hand = hands[i];
    for (let j = 0; j < connections.length; j++) {
      let pointAIndex = connections[j][0];
      let pointBIndex = connections[j][1];
      let pointA = hand.keypoints[pointAIndex];
      let pointB = hand.keypoints[pointBIndex];
      stroke(255, 0, 0);
      strokeWeight(2);
      line(pointA.x, pointA.y, pointB.x, pointB.y);
    }
  }

  // Draw all the tracked hand points
  for (let i = 0; i < hands.length; i++) {
    let hand = hands[i];
    for (let j = 0; j < hand.keypoints.length; j++) {
      let keypoint = hand.keypoints[j];
      fill(0, 255, 0);
      noStroke();
      circle(keypoint.x, keypoint.y, 10);
    }
  }

  pop();
}

// A function to draw a circle using the tip of the thumb and the index
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/handPose-parts-p5-2.0/sketch.js
function drawPinch() {
  push();

  // If there is at least one hand
  if (hands.length > 0) {
    // Find the index finger tip and thumb tip
    let finger = hands[0].index_finger_tip;
    let thumb = hands[0].thumb_tip;

    // Compute the coordinate of the center between thumb and index
    let centerX = (finger.x + thumb.x) / 2;
    let centerY = (finger.y + thumb.y) / 2;
    // Calculate the pinch "distance" between finger and thumb
    let pinch = dist(finger.x, finger.y, thumb.x, thumb.y);

    // This circle's size is controlled by a "pinch" gesture
    fill(0, 255, 0, 200);
    stroke(0);
    strokeWeight(2);
    circle(centerX, centerY, pinch);
  }

  pop();
}

// Call this function to start and stop detection
// Source: https://github.com/ml5js/ml5-next-gen/blob/main/examples/handPose-detect-start-stop-p5-2.0/sketch.js
function toggleDetection() {
  if (isDetecting) {
    handPose.detectStop();
    isDetecting = false;
  } else {
    handPose.detectStart(video, gotHands);
    isDetecting = true;
  }
}


function gotHands(results) {
    // Save the output to the hands variable
  hands = results;
}

function keyPressed() {
  if (key === " ") {
    mode = (mode + 1) % 3;
  } else if (key === "s") {
    toggleDetection();
  }
}

// This time, using a click to display the hand object
function mousePressed() {
  console.log("Hand detection object:");
  console.log(hands);
}

// IDEA: one thing that could be done, to familiarise yourself with the landmarks and
//       the geometry of the hands, would be to draw lines between the landmarks, to
//       create a silhouette of a hand, as seen here for instance:
//       https://github.com/tensorflow/tfjs-models/tree/master/handpose#mediapipe-handpose
