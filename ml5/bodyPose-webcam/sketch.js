/*
  Data and machine learning for creative practice (DMLCP)

  Body Pose with PoseNet

  Originally from: The Coding Train / Daniel Shiffman
  (Tutorials for v0)
  https://thecodingtrain.com/Courses/ml5-beginners-guide/7.1-posenet.html
  https://youtu.be/OIo-DIOkNVg
  https://editor.p5js.org/codingtrain/sketches/ULA97pJXR
  An other long tutorial with experiments:
  https://www.youtube.com/watch?v=931bqqpQqvI

  Reference here: https://docs.ml5js.org/#/reference/bodypose
*/

let video,
    bodyPose,
    poses,
    connections;

function preload() {
  bodyPose = ml5.bodyPose();
  // 'MoveNet' by default, try also : 'BlazePose'
  // see: https://docs.ml5js.org/#/reference/bodypose?id=ml5bodypose
}

function setup() {

  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();

  // Start detecting poses in the webcam video
  bodyPose.detectStart(video, gotPoses);

  // Get the skeleton connection information
  connections = bodyPose.getSkeleton();

}

function gotPoses(results) {

  // `results` is an array (the model can detect more than one person!)
  poses = results;

}

function draw() {
  image(video, 0, 0);

  // IDEA: like in previous sketches, there is no obligation to display the
  //       video, and you could for instance imagine a blank canvas where a few
  //       points from the body are used to control the position, or other
  //       parameters (the size?) of text that is displayed on the screen, like
  //       in the Google and Bill T. Jones collaboration here:
  //       https://experiments.withgoogle.com/billtjonesai

  drawPose(); // We call our function to draw the keypoints
}

// Dan Shiffman's function drawing various kinds of dots in various keypoints
function drawPose() {

  if (!poses) return; // only execute if we have poses detected

  // Draw the skeleton connections
  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];

    for (let j = 0; j < connections.length; j++) {
      let pointAIndex = connections[j][0];
      let pointBIndex = connections[j][1];
      let pointA = pose.keypoints[pointAIndex];
      let pointB = pose.keypoints[pointBIndex];

      // Only draw a line if we have confidence in both points
      if (pointA.confidence > 0.1 && pointB.confidence > 0.1) {
        stroke(255, 0, 0);
        strokeWeight(1);
        line(pointA.x, pointA.y, pointB.x, pointB.y);
      }
    }
  }

  // Iterate through all the poses
  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
    // Iterate through all the keypoints for each pose
    for (let j = 0; j < pose.keypoints.length; j++) {
      let keypoint = pose.keypoints[j];
      // Only draw a circle if the keypoint's confidence is greater than 0.1
      if (keypoint.confidence > 0.1) {
        fill(0, 255, 0);
        noStroke();
        circle(keypoint.x, keypoint.y, 5);
      }
    }
  }

  // IDEA: you could instead use the distance between, say, the nose
  //       and one wrist, that would allow a user to control the size of a
  //       figure!, or to make something happen if they touch their noses!
  //       In this project, this idea is used to allow users to scream silently!
  //       https://x.com/JordanneChan/status/1483988494032805895

  // IDEA: going further with this, you have no obligation to work with the entire
  //       skeleton, and perhaps you might want to use only the two wrists, or the
  //       two knees, or less obvious combinations (eye and hip?, wrist and knee?)

}

// A click of the mouse logs the pose & connections!
function mousePressed() {
  if (poses) {
    console.log('all poses:');
    console.log(poses);
    console.log('connections:');
    console.log(connections);
    console.log('-----------------------------');
  } else {
    console.log("No pose detected yet");
  }
}

