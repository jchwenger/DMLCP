/*
 * 👋 Hello! This is an ml5.js example made and shared with ❤️.
 * Learn more about the ml5.js project: https://ml5js.org/
 * ml5.js license and Code of Conduct: https://github.com/ml5js/ml5-next-gen/blob/main/LICENSE.md
 *
 * This example demonstrates separating people from the background with ml5.bodySegmentation.
 *
 * Documentation: https://docs.ml5js.org/#/reference/body-segmentation
 * Sources: https://github.com/ml5js/ml5-next-gen/tree/main/examples/bodySegmentation-mask-person-p5-2.0
 *          https://github.com/ml5js/ml5-next-gen/tree/main/examples/bodySegmentation-mask-background-p5-2.0
 */

let bodySegmentation;
let video;
let segmentation;

let options = {
  maskType: "background", // try "person"
};

async function setup() {
  createCanvas(640, 480);

  bodySegmentation = await ml5.bodySegmentation("SelfieSegmentation", options);

  // Create the video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  bodySegmentation.detectStart(video, gotResults);
}

function draw() {
  // the background controls the color of the mask!
  background(0, 0, 255);
  if (segmentation) {
    video.mask(segmentation.mask);
    image(video, 0, 0);
  }
}

// callback function for body segmentation
function gotResults(result) {
  segmentation = result;
}

function mousePressed() {
  console.log("Segmentation result:");
  console.log(segmentation);
}
