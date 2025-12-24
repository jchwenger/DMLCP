/*
  Data and machine learning for creative practice (DMLCP)
  Week 6 Image-to-Image Translation

  Interactive Pix2pix example with p5.js using callback functions
  This uses a pre-trained model on Pikachu images

  The pre-trained Edges2Pikachu model is trained on 256x256 images
  So the input images can only be 256x256 or 512x512, or multiple of 256

  Original sketch: https://editor.p5js.org/ml5/sketches/Pix2Pix_callback
  (broken-ish?, needs an earlier version of ml5.js, e.g. 0.6.0)
  Reference here: https://archive-docs.ml5js.org/#/reference/pix2pix
*/

const SIZE = 256;
let inputImg,
    inputCanvas,
    outputContainer,
    statusMsg,
    pix2pix,
    clearBtn,
    modelReady = false,
    isTransfering = false;

function setup() {
  // Create a canvas
  inputCanvas = createCanvas(SIZE, SIZE);
  inputCanvas.class('border-box').parent('canvasContainer');

  // Display initial input image
  inputImg = loadImage('images/input.png', drawImage);

  // Selcect output div container
  outputContainer = select('#output');
  statusMsg = select('#status');

  // Select 'clear' button html element
  clearBtn = select('#clearBtn');
  // Attach a mousePressed event to the 'clear' button
  clearBtn.mousePressed(function() {
    clearCanvas();
  });

  // Set stroke to black
  stroke(0);
  pixelDensity(1);

  // Create a pix2pix method with a pre-trained model
  // available styles:
  const modelURLs = {
    "edges2pikachu": "https://rawcdn.githack.com/jchwenger/ml5-data-and-models/f3ec7c10db635d1e591d587e4f053395f6d8fb85/models/pix2pix/edges2pikachu_AtoB.pict",
    "edges2cats": "https://rawcdn.githack.com/jchwenger/ml5-data-and-models/f3ec7c10db635d1e591d587e4f053395f6d8fb85/models/pix2pix/edges2cats_AtoB.pict",
    "edges2shoes": "https://rawcdn.githack.com/jchwenger/ml5-data-and-models/f3ec7c10db635d1e591d587e4f053395f6d8fb85/models/pix2pix/edges2shoes_AtoB.pict",
    "edges2handbags": "https://rawcdn.githack.com/jchwenger/ml5-data-and-models/f3ec7c10db635d1e591d587e4f053395f6d8fb85/models/pix2pix/edges2handbags_AtoB.pict",
    "facades": "https://rawcdn.githack.com/jchwenger/ml5-data-and-models/f3ec7c10db635d1e591d587e4f053395f6d8fb85/models/pix2pix/facades_BtoA.pict"
  }
  // urls found here: https://github.com/khanniie/mldraw/blob/cd1d58a73f659ba494fff06a765446d0d9aa4931/frontend/src/local-models.ts#L9
  // originally from these two repos:
  // https://github.com/affinelayer/pix2pix-tensorflow-models (no longer available)
  // https://github.com/ml5js/ml5-data-and-models/tree/main/models/pix2pix
  // Create a pix2pix method with a pre-trained model
  const style = "edges2pikachu";
  pix2pix = ml5.pix2pix(modelURLs[style], modelLoaded);
}

// Draw on the canvas when mouse is pressed
function draw() {
  if (mouseIsPressed) {
    line(mouseX, mouseY, pmouseX, pmouseY);
  }
}

// Whenever mouse is released, transfer the current image if the model is loaded and it's not in the process of another transformation
function mouseReleased() {
  if (modelReady && !isTransfering) {
    transfer()
  }
}

// A function to be called when the models have loaded
function modelLoaded() {
  // Show 'Model Loaded!' message
  statusMsg.html('Model Loaded!');

  // Set modelReady to true
  modelReady = true;

  // Call transfer function after the model is loaded
  transfer();

}

// Draw the input image to the canvas
function drawImage() {
  image(inputImg, 0, 0);
}

// Clear the canvas
function clearCanvas() {
  background(255);
}

function transfer() {
  // Set isTransfering to true
  isTransfering = true;

  // Update status message
  statusMsg.html('Applying Style Transfer...!');

  // Select canvas DOM element
  const canvasElement = select('canvas').elt;

  // Apply pix2pix transformation
  pix2pix.transfer(canvasElement, function(err, result) {
    if (err) {
      console.log(err);
    }
    if (result && result.src) {
      // Set isTransfering back to false
      isTransfering = false;
      // Clear output container
      outputContainer.html('');
      // Create an image based result
      createImg(result.src).class('border-box').parent('output');
      // Show 'Done!' message
      statusMsg.html('Done!');
    }
  });
}
