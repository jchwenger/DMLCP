/*
 * 👋 Hello! This is an ml5.js example made and shared with ❤️.
 * Learn more about the ml5.js project: https://ml5js.org/
 * ml5.js license and Code of Conduct: https://github.com/ml5js/ml5-next-gen/blob/main/LICENSE.md
 *
 * This example demonstrates segmenting a person by body parts with ml5.bodySegmentation.
 *
 * Documentation: https://docs.ml5js.org/#/reference/body-segmentation
 * Sources: https://github.com/ml5js/ml5-next-gen/tree/main/examples/bodySegmentation-mask-body-parts-p5-2.0
 *          https://github.com/ml5js/ml5-next-gen/tree/main/examples/bodySegmentation-select-body-parts-p5-2.0
 */

let bodySegmentation;
let video;
let segmentation;
let mode = 0; // press space to cycle through, display all parts or only a selected part

let parts = [];
let partsArray = [];
let selectedPart = 12; // FRONT_TORSO

let options = {
  maskType: "parts",
};

async function setup() {
  createCanvas(640, 480);

  bodySegmentation = await ml5.bodySegmentation("BodyPix", options);

  // Create the video
  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  bodySegmentation.detectStart(video, gotResults);
  parts = bodySegmentation.getPartsId();
  partsArray = Object.keys(parts);
}

function draw() {
  background(255);
  image(video, 0, 0);

  switch(mode) {
    case 0:
      drawBodyParts();
      break;
    case 1:
      drawSelectedBodyParts();
      break;
  }
}

function drawBodyParts() {
  push();
  if (segmentation) {
    image(segmentation.mask, 0, 0, width, height);
  }
  pop();
}

function drawSelectedBodyParts() {
  push();
  image(video, 0, 0);
  if (segmentation) {
    let gridSize = 5;
    for (let x = 0; x < video.width; x += gridSize) {
      for (let y = 0; y < video.height; y += gridSize) {
        if (segmentation.data[y * video.width + x] == selectedPart) {
          fill(255, 0, 0);
          noStroke();
          circle(x, y, gridSize);
        }
      }
    }
  }
  pop();
}

// callback function for body segmentation
function gotResults(result) {
  segmentation = result;
}

function keyPressed() {
  if (key === " ") {
    mode = (mode + 1) % 2;
    console.log(`Mode is now ${mode}`);
  } else if (key === "p" && mode === 1) {
    selectedPart = (selectedPart + 1) % partsArray.length;
    console.log(`Selected part: ${selectedPart}, ${partsArray[selectedPart]}`);
  }
}

function mousePressed() {
  console.log("Segmentation result:");
  console.log(segmentation);
  console.log("Parts object:");
  console.log(parts);
  console.log("Parts array:");
  console.log(partsArray);
}
