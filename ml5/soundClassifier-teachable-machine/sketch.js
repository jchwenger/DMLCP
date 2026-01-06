/*
  Data and machine learning for creative practice (DMLCP)

  Sound classifier via models trained with Teachable Machines
  originally from https://github.com/ml5js/Intro-ML-Arts-IMA-F21
  More rececent course: https://github.com/ml5js/Intro-ML-Arts-IMA-F24

  Reference (deprecated): https://archive-docs.ml5js.org/#/reference/sound-classifier
  Updated to this (works like imageClassifier): https://docs.ml5js.org/#/reference/image-classifier
  See also this: https://docs.ml5js.org/#/reference/sound-classifier

  Note: when I tested it, the Teachable Machines interface didn't
  work so well with Firefox (OK with Chrome).
*/

let classifier,
    hasLogged = false, // log only once
    label = "listening";

let results;

// Label (start by showing listening)
// IDEA: currently, the sketch will never go back to the original string once
//       one detection has occurred. Maybe you would want 'listening' to be a
//       default state the sketch returns to?

// Teachable Machine model URL:
const soundModelURL = 'https://teachablemachine.withgoogle.com/models/h3p9R41J/'; // hand clap
const instructions = 'Clap your hands!';

// Other model
// const soundModelURL = 'https://teachablemachine.withgoogle.com/models/ueOfZRiLS/'; // 'blah', 'ouh', 'OK'
// const instructions = 'Say "blah", "ouh", or "OK"';

// IDEA: this would allow you to learn commands that activate certain actions or effects in
//       the sketch, for instance, the creation of certain shapes?
// IDEA: it is possible to upload samples to Teachable Machines, perhaps you could try this
//       with songs from two different bands, and see if it recognises a song it was not trained
//       on? (You might hit a wall there, especially if you don't have a lot of training data,
//       but it would be quite interesting to try it out!)

function preload() {
  // Load the model
  classifier = ml5.soundClassifier(soundModelURL + 'model.json');
}

function setup() {
  createCanvas(320, 240);
  createP(instructions);
  // Start classifying
  // The sound model will continuously listen to the microphone
  classifier.classifyStart(gotResult);
}

function draw() {
  background(0);
  // Draw the label in the canvas
  fill(255);
  textSize(32);
  textAlign(CENTER, CENTER);
  text(label, width / 2, height / 2);
}


// The model recognizing a sound will trigger this event
function gotResult(results) {

  if (!results) return; // avoid error before any detection is made

  // The results are in an array ordered by confidence.
  if (!hasLogged) {
    console.log(results);
    hasLogged = true;
  }
  label = results[0].label;

}
