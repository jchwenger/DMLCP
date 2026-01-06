/*
  Data and machine learning for creative practice (DMLCP)

  Handpose webcam demo

  Originally from here (deprecated): https://editor.p5js.org/ml5/sketches/Handpose_Webcam
  Reference here: https://docs.ml5js.org/#/reference/handpose
  And other examples: https://docs.ml5js.org/#/reference/handpose?id=examples
*/

let video,
    handPose,
    hands = [];

function preload() {
  handPose = ml5.handPose();
}

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
}

function draw() {

  // if we have the video access yet, draw it
  if (video) {
    image(video, 0, 0);
  }

  // Start detecting hands from the webcam video
  handPose.detectStart(video, gotHands);

  // IDEA: like in previous sketches, there is no obligation to display the video,
  //       and you could for instance imagine a blank canvas where points from the
  //       hand, can be used in artistic ways! To draw or influence shapes on
  //       the canvas in real time!
  // IDEA: (advanced) in this vein, it might be possible to train a sound model
  //       on Teachable Machine using different hand poses, and then combine the
  //       local sound sketch with this one, where the landmarks control
  //       animations and sound!

  drawHand();
}

function gotHands(results) {
    // Save the output to the hands variable
  hands = results;
}

// This time, using a click to display the hand object
function mousePressed() {
  console.log(hand);
}

function drawHand() {
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

// IDEA: one thing that could be done, to familiarise yourself with the landmarks and
//       the geometry of the hands, would be to draw lines between the landmarks, to
//       create a silhouette of a hand, as seen here for instance:
//       https://github.com/tensorflow/tfjs-models/tree/master/handpose#mediapipe-handpose
