/*
  Data and machine learning for creative practice (DMLCP)
  Week 6 Image-to-Image Translation

  Style transfer on a webcam

  Adapted from Joe McAlister's version

  Original sketch (broken?): https://editor.p5js.org/ml5/sketches/StyleTransfer_Video
  (↑ broken unless you go into the html file and downgrade ml5 to e.g. 0.6.0)
  Reference here: https://archive-docs.ml5js.org/#/reference/style-transfer
*/

let styleModel,
    video,
    img,
    graphics;

function setup() {
  createCanvas(320, 240);

  // load webcam and hide it, videoLoaded callback for when its available
  video = createCapture(VIDEO, videoLoaded);
  video.size(320, 240);
  video.hide();

  // create graphics, you can image this like a second canvas, only invisible
  graphics = createGraphics(320, 240);
}

function videoLoaded(stream) {
  // load in the style transfer model, using the ml5 repository:
  // https://github.com/ml5js/ml5-data-and-models/tree/main/models/style-transfer
  // (you could also clone the repo and copy the model folders into a `models` folder inside the sketch
  // avilable styles:
  // "la_muse", "mathura", "matilde_perez", "matta", "rain_princess",
  // "scream", "udnie", "wave", "wreck"
  const style = "wave";
  styleModel = ml5.styleTransfer(`https://raw.githubusercontent.com/ml5js/ml5-data-and-models/master/models/style-transfer/${style}`, modelLoaded);
  // IDEA: (An inspiration for live model switching can be found here:
  //       https://github.com/ml5js/ml5-data-and-models/)
}

function modelLoaded() {
  // model loaded
  console.log("Model loaded");

  // start the transfer of style
  transferStyle();
}

function transferStyle() {
  // we transfer based on graphics, graphics contains a scaled down video feed
  styleModel.transfer(graphics, function(err, result) {
    let tempDOMImage = createImg(result.src).hide();
    img = tempDOMImage;
    tempDOMImage.remove(); // remove the temporary DOM image

    // recursively call function so we get live updates
    transferStyle();
  });
}

function draw(){
  // Switch between showing the raw camera or the style
  if (img) {
    image(img, 0, 0, 320, 240);
  }

  // this puts the video feed into the invisible graphics canvas
  graphics.image(video, 0, 0, 320, 240);
}
