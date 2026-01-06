/*
  Data and machine learning for creative practice (DMLCP)

  Image classification demo

  Reference here: https://docs.ml5js.org/#/reference/image-classifier
*/

let classifier,
    result_string = "Predicting...",
    img;

// Instantiate our model and load our image *before* setup
// (this is the role of `preload`, to be run before `setup`)
function preload() {

  classifier = ml5.imageClassifier("MobileNet"); // also: DarkNet , DarkNet-Tiny, DoodleNet

  // load in our classifier from the internet, downloading the model if need be
  // (by default the model returns the top 3 predictions, for more pass the option
  // {topk: 5}, or any number up to 1000, as a second argument!)
  // See: https://docs.ml5js.org/#/reference/image-classifier?id=ml5imageclassifier

  img = loadImage("images/labrador.jpg");

  // load our dog image; try the australian-labradoodle-guide.jpg, does it work well?
  // IDEA: import, or even create, your own images, see what the model is able to recognise
  // IDEA: how about implementing more of an interface to load images dynamically?
  //       This could be, for instance, loading all the images in the `images/` folder,
  //       and allowing the user to switch between images. Or even use an API to pull images
  //       from the Internet?

}

function setup() {
  createCanvas(img.width, img.height + 50);

  // IDEA: here, instead of using an existing image, you could *create one*
  //       using createGraphics, and see what the network recognises as an
  //       image or not? Try with just random colours everywhere, for instance,
  //       how does the network react?
  // IDEA: more advanced, you could imagine an interface where by pressing a
  //       key some randomness, or various shapes, are *added* to an existing
  //       image (or even something where the user can draw on the image) –
  //       then, the network would be requested to classify the new image (you
  //       probably need to import the image into a graphics object you can
  //       draw onto, or extract the canvas itself into an image object, that
  //       can be passed to the network). The interesting thing is that this
  //       would allow people to see how the new patterns gradually degrade the
  //       network's performance!

                        // ↓ the callback, see below
  classifier.classify(img, gotResults);

  // Call classify on the image
  // We do this in setup as we only want to do this once. We pass in a function
  // `gotResults` that will be called once the classifier has identified what
  // our image is. Note that we pass the name of the function, as an argument
  // to the function. In JavaScript functions can be treated as any other
  // variable. So they can be assigned, e.g. I could do:
  //
  // let f = gotResults;
  //
  // and then `f` and `gotResults` would represent exactly the same function.
  // Note that we do not add the parentheses at the end, it is `gotResults` and
  // not `gotResults()`. that means we are treating the function as a
  // *variable*, while adding the `()` at the end, with eventual arguments
  // would actually *call* the function (i.e. exectute it) and gives us back
  // whatever that the function returns. A function passed in in this manner is
  // commonly referred to as a *callback*, and this is quite common to see in
  // JavaScript.

}

function draw() {
  background(255);

  // Draw the dog
  image(img, 0, 0, img.width, img.height);

  // Draw the prediction, this is waiting text until gotResults is called
  fill(0);
  textSize(15);
  textAlign(CENTER);
  text(result_string, 0, height - 40, width, 40);
}

// The callback function. This function is called when the image classifier has
// decided on what the image contains, or has thrown an error.
function gotResults(results) {

  // Log our results to the console so we can see them
  console.log("We think we know what this is...");
  console.log(results);

  // Form a string to contain the results – here we use the backtick method of
  // embedding variables in strings. We use Math.ceil to round up the decimal
  // to the closest whole number
  const confidence_percentage = (Math.ceil(results[0].confidence * 100));
  result_string = `This is a ${results[0].label}!\n(I'm ${confidence_percentage}% confident.)`;

  // IDEA: note that here we use *only the top detection* (results[0])! But
  //       you could imagine using more than the top one, and display them in
  //       some way. This would be especially interesting if you are able to
  //       gradually modify an image, or test many different images, to see
  //       what other predictions the model is doing. Have look inside the
  //       results array, to see how it works:
  // console.log(results)

}
