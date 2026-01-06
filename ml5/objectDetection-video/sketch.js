/*
 * 👋 Hello! This is an ml5.js example made and shared with ❤️.
 * Learn more about the ml5.js project: https://ml5js.org
 * ml5.js license and Code of Conduct: https://github.com/ml5js/ml5-next-gen/blob/main/LICENSE.md
 *
 * Source: https://github.com/ml5js/ml5-next-gen/tree/main/examples/objectDetection-video-p5-2.0
 * Documentation: https://docs.ml5js.org/#/reference/object-detection
 */

let video;
let detector;
let detections = [];

async function setup() {
  detector = await ml5.objectDetection("cocossd");

  createCanvas(640, 480);
  
  // Load and loop the video for object detection
  // TODO: check these pesky autoplay issues in browsers a bit more deeply
  video = await createVideo('ball_lifting.mp4'); // video sized 640 x 480
  // Play the video when the user clicks on it.
  video.hide();
  video.loop();
  
  detector.detectStart(video, gotDetections);
}

function draw(){
  image(video, 0, 0); // draw video frame

  for (let i = 0; i < detections.length; i++) {
    let detection = detections[i];

    let x = detection.x;
    let y = detection.y;
    let w = detection.width;
    let h = detection.height;

    stroke(0, 255, 0);
    strokeWeight(4);
    noFill();
    rect(x, y, w, h);

    noStroke();
    fill(255);
    textSize(18);
    text(detection.label, x + 5, y + 20);
  }
}

// Callback function is called each time the object detector finishes processing a frame.
function gotDetections(results) {
  // Update detections array with the new results
  detections = results;  
}
