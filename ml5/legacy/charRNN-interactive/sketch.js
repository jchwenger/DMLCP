/*
  Data and machine learning for creative practice (DMLCP)
  Week 7 Text & Sequences

  Interactive LSTM Text Generation Example using p5.js from ml5.js

  Original sketch: https://github.com/ml5js/ml5-library/tree/main/examples/p5js/CharRNN/CharRNN_Interactive
  Reference here: https://archive-docs.ml5js.org/#/reference/charrnn
*/

let charRNN;
let textInput;
let tempSlider;
let lengthSlider;
let runningInference = true;

let generated = false;
let last;

let original;

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
  lengthSlider = select('#lenSlider');
  tempSlider = select('#tempSlider');

  // Run "changing" anytime something changes
  textInput.input(changing);
  lengthSlider.input(changing);
  tempSlider.input(changing);

  // Check every so often if we should generate something new
  setInterval(checkGenerate, 500);
}

function modelReady() {
  select('#status').html('Model Loaded');
  runningInference = false;
}

// Has 500 milliseconds passed since the last time a change was made?
function checkGenerate() {
  const passed = millis() - last;
  if (passed > 500 && !generated) {
    generate();
    generated = true;
  }
}

// Update the time
function changing() {
  generated = false;
  last = millis();
}

// Generate new text!
function generate() {
  // Grab the original text
 original = textInput.value();
  // Make it to lower case
  const txt = original.toLowerCase();

  // prevent starting inference if we've already started another instance
  // or if there is no prompt
  // TODO: is there better JS way of doing this?
  if (!runningInference && txt.length > 0) {
    runningInference = true;

    // Update the status log
    select('#status').html('Generating...');

    // Update the length and temperature span elements
    select('#length').html(lengthSlider.value());
    select('#temperature').html(tempSlider.value());

    // Here is the data for the LSTM generator
    const data = {
      seed: txt,
      temperature: tempSlider.value(),
      length: lengthSlider.value(),
    };

    // Generate text with the charRNN
    charRNN.generate(data, gotData);
  }
}

// Update the DOM elements with typed and generated text
function gotData(err, result) {
  runningInference = false;
  if (err) {
    console.error(err);
    return;
  }
  select('#status').html('Ready!');
  select('#original').html(original);
  select('#prediction').html(result.sample);
  // IDEA: one clear thing that could be done here is to modify this sketch so that
  //       e.g. the text box is not used only for initial text, but to continue the
  //       text! Perhaps, once some text has been generated, you could remove the
  //       content of the text box, and when the user writes something in it again,
  //       it will be considered *a continuation* of the network output – then what you
  //       would need to do is to use the contents of #result as the seed, perhaps cropping
  //       it to a certain length (only the end of the text)?
}

// IDEA: in this sketch, the text is stored in HTML, but you could also use the
//       text() function in p5.js to display the results and combine that with
//       other animations!

