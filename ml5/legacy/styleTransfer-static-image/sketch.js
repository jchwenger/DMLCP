/*
  Data and machine learning for creative practice (DMLCP)
  Week 6 Image-to-Image Translation
  
  Style transfer on a static image
  
  Adapted from Joe McAlister's version
  
  Original sketch : https://editor.p5js.org/ml5/sketches/StyleTransfer_Image
  (↑ broken unless you go into the html file and downgrade ml5 to e.g. 0.6.0)
  Reference here: https://archive-docs.ml5js.org/#/reference/style-transfer
*/

let styleModel,
    img,
    originalImg,
    canvas;

function preload() {
  // load our transfer image
  img = loadImage("img/patagonia.jpg");
  originalImg = img; // this is used to restore later
}

function setup() {
  // create canvas, we assign it to the canvas variable so we can access it later
  canvas = createCanvas(500, 500);

  // load in the style transfer model, using the ml5 repository:
  // https://github.com/ml5js/ml5-data-and-models/tree/main/models/style-transfer
  // (you could also clone the repo and copy the model folders into a `models` folder inside the sketch
  // avilable styles:
  // "la_muse", "mathura", "matilde_perez", "matta", "rain_princess",
  // "scream", "udnie", "wave", "wreck"
  const style = "wave";
  styleModel = ml5.styleTransfer(`https://raw.githubusercontent.com/ml5js/ml5-data-and-models/master/models/style-transfer/${style}`, modelLoaded);
  // IDEA: you could instantiate more than one model at the same time!
  //       (and perhaps create a pipeline where you give the output of
  //       one model to the next...?)
}

function draw() {
  background(0);

  // draw our image
  image(img, 0, 0, width, height);
}

// this function is called when either model is loaded
function modelLoaded() {
  // Check if the model are loaded
  if (styleModel.ready) {
    console.log("Model is ready!");
  }
}

function keyPressed() {
  if (key == "1") {
    // we should transfer into style 1
    img = originalImg; // switch back to original so we can prevent recursive application
    styleModel.transfer(canvas, function(err, result) {
      let tempDOMImage = createImg(result.src).hide();
      img = tempDOMImage;
    });

  }

  if (key == "c") {
    // we should transfer back
    img = originalImg;
  }
}
