/*
  Data and machine learning for creative practice (DMLCP)

  Sound classifier via models trained with Teachable Machines
  originally from https://github.com/ml5js/Intro-ML-Arts-IMA-F21
  More rececent course: https://github.com/ml5js/Intro-ML-Arts-IMA-F24

  Documentation: https://docs.ml5js.org/#/reference/sound-classifier
  Source: https://github.com/ml5js/ml5-next-gen/tree/main/examples/soundClassifier-teachable-machine-p5-2.0

  Note: when I tested it, the Teachable Machines interface didn't
  work so well with Firefox (OK with Chrome).
*/

let classifier,
    hasLogged = false, // log only once
    predictedSound = "";

let results;

// Label (start by showing listening)
// IDEA: currently, the sketch will never go back to the original string once
//       one detection has occurred. Maybe you would want 'listening' to be a
//       default state the sketch returns to?

// Teachable Machine model URL:
const soundModelURL = "https://teachablemachine.withgoogle.com/models/FvsFiSwHW/"; // hand clap & whistle

// IDEA: this would allow you to learn commands that activate certain actions or effects in
//       the sketch, for instance, the creation of certain shapes?
// IDEA: it is possible to upload samples to Teachable Machines, perhaps you could try this
//       with songs from two different bands, and see if it recognises a song it was not trained
//       on? (You might hit a wall there, especially if you don't have a lot of training data,
//       but it would be quite interesting to try it out!)


async function setup() {

  // Load the model
  classifier = await ml5.soundClassifier(soundModelURL);

  createCanvas(650, 450);
  textAlign(CENTER, CENTER);
  textSize(32);

  // Start classifying
  // The sound model will continuously listen to the microphone
  classifier.classifyStart(gotResult);
}

function draw() {
  background(250);

  // Update canvas according to classification results
  if (predictedSound == "Background Noise" || predictedSound == "") {
    fill(0);
    textSize(64);
    text("clap 👏 or whistle 🎵 ", width / 2, height / 2);
  } else if (predictedSound == "Clap") {
    background(231, 176, 255);
    textSize(128);
    text("👏", width / 2, height / 2);
  } else if (predictedSound == "whistle") {
    background(255, 242, 143);
    textSize(128);
    text("🎵", width / 2, height / 2);
  }

}


// The model recognizing a sound will trigger this event
function gotResult(results) {
  // The results are in an array ordered by confidence.
  console.log(results);
  // Store the first label
  predictedSound = results[0].label;

}
