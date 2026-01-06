/*
  Data and machine learning for creative practice (DMLCP)
  Week 7 Text & Sequences

  Stateful LSTM Text Generation Example using p5.js

  Modified sketch from the ml5.js library

  Original sketch: https://github.com/ml5js/ml5-library/tree/main/examples/p5js/CharRNN/CharRNN_Text_Stateful
  Reference here: https://archive-docs.ml5js.org/#/reference/charrnn

  Acknowledgements
  ----------------

  This work was supported through a residency at the Frank-Ratchye STUDIO for
  Creative Inquiry (http://studioforcreativeinquiry.org/)at Carnegie Mellon
  University, Pittsburgh, October 2018, with additional support from the Sylvia
  and David Steiner Speaker Series.

*/

let charRNN;
let textInput;
let tempSlider;
let startBtn;
let resetBtn;
let singleBtn;
let generating = false;

const canvasHeight = 100;

function setup() {
  noCanvas();
  // Create the LSTM Generator passing it the model directory
  const baseURL = 'https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/charRNN/';
  const modelName = 'woolf';
  // Also available: bolano             // https://en.wikipedia.org/wiki/Roberto_Bola%C3%B1o
                  // charlotte_bronte   // https://en.wikipedia.org/wiki/Charlotte_Bront%C3%AB
                  // darwin             // https://en.wikipedia.org/wiki/Charles_Darwin
                  // dubois             // https://en.wikipedia.org/wiki/W._E._B._Du_Bois
                  // hemingway          // https://en.wikipedia.org/wiki/Ernest_Hemingway
                  // shakespeare        // https://en.wikipedia.org/wiki/William_Shakespeare
                  // woolf              // https://en.wikipedia.org/wiki/Virginia_Woolf
                  // zora_neale_hurston // https://en.wikipedia.org/wiki/Zora_Neale_Hurston
  charRNN = ml5.charRNN(`${baseURL}${modelName}/`, modelReady);
  // IDEA: add an UI functionality that allows the user to change the model as the sketch is running
  //       (that has been implemented here, not in p5.js: https://github.com/cvalenzuela/Selected_Stories)

  // Grab the DOM elements
  textInput = select('#textInput');
  tempSlider = select('#tempSlider');
  startBtn = select('#start');
  resetBtn = select('#reset');
  singleBtn = select('#single');

  // DOM element events
  startBtn.mousePressed(generate);
  resetBtn.mousePressed(resetModel);
  singleBtn.mousePressed(predict);
  tempSlider.input(updateSliders);
}

function windowResized() {
  resizeCanvas(windowWidth, canvasHeight);
}

// Update the slider values
function updateSliders() {
  select('#temperature').html(tempSlider.value());
}

async function modelReady() {
  select('#status').html('Model Loaded');
  resetModel();
}

function resetModel() {
  charRNN.reset();
  const seed = select('#textInput').value();
  charRNN.feed(seed);
  select('#result').html(seed);
}

function generate() {
  if (generating) {
    generating = false;
    startBtn.html('Start');
  } else {
    generating = true;
    startBtn.html('Pause');
    loopRNN();
  }
}

async function loopRNN() {
  while (generating) {
    await predict();
  }
}

async function predict() {
  const par = select('#result');
  const temperature = tempSlider.value();
  const next = await charRNN.predict(temperature);

  await charRNN.feed(next.sample);

  // Respect new lines!
  // If the sample is a new line, use an actual html newline...
  if (next.sample.match(/\n/)) {
    // console.log("newline!", next.sample);
    par.html(par.html() + '<br>');
  } else {
    par.html(par.html() + next.sample);
  }
}

// IDEA: just like in the interactive example, here one might modify the code
//       to allow the text box not just to be the beginning, but also to allow
//       the user to inject code within the text? Then you would need to feed
//       the content of the text box into the net, before allowing for more
//       generation

// IDEA: in this sketch, the text is stored in HTML, but you could also use the
//       text() function in p5.js to display the results and combine that with
//       other animations!

