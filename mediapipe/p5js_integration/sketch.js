/* Sketch by Becky Aston
 * https://editor.p5js.org/beckyaston/sketches/431uCva8m
 *
 *  p5.js + Google MediaPipe 
 *  Allows tracking of landmarks on hands, bodies, and/or faces.
 *  See https://mediapipe-studio.webapps.google.com/home
 *  Code adapted from:
 *  Golan Levin: https://editor.p5js.org/golan/sketches/0yyu6uEwM
 *  Orr Kislev: https://editor.p5js.org/orrkislev/sketches/wwLqrnVDt
 *
 *
 *  Works best with hands OR body OR face — but lags when more than one is
 *  enabled. NOTE: don't have the doAcquireHandGesture and hand
 *  doAcquireHandLandmarks both set to true in the tracking Config (see below),
 *  Gesture detection will get the landmarks & you will otherwise slow the
 *  frame-rate down for duplicate work.
 *  ---------------------------------------------------------------------------
 */

// Don't change the names of these global variables.
let myHandLandmarker;
let myGestureRecognizer;
let myPoseLandmarker;
let myFaceLandmarker;
let handLandmarks;
let gestureResults;
let poseLandmarks;
let faceLandmarks;
let myCapture;
let lastVideoTime = -1;

// For features you want, set to true; set false the ones you don't.
// Works best with just one or two sets of landmarks.
const trackingConfig = {
  doAcquireHandLandmarks: true, // don't run this if you run gestures
  doAcquireHandGesture: !true, //comes with landmarks & gestures
  doAcquirePoseLandmarks: !true,
  doAcquireFaceLandmarks: !true,
  doAcquireFaceMetrics: !true, // use with doAcquireFaceLandmarks
  poseModelLiteOrFull: "lite", /* "lite" (3MB) or "full" (6MB) */
  cpuOrGpuString: "GPU", /* "GPU" or "CPU" */
  maxNumHands: 2,
  maxNumPoses: 1,
  maxNumFaces: 1,
};


//------------------------------------------
async function preload() {
  const mediapipe_module = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.js');
  
  HandLandmarker = mediapipe_module.HandLandmarker;
  GestureRecognizer = mediapipe_module.GestureRecognizer;
  PoseLandmarker = mediapipe_module.PoseLandmarker;
  FaceLandmarker = mediapipe_module.FaceLandmarker;
  FilesetResolver = mediapipe_module.FilesetResolver;
  
  
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // Hand Landmark Tracking:
  // https://codepen.io/mediapipe-preview/pen/gOKBGPN
  // https://mediapipe-studio.webapps.google.com/studio/demo/hand_landmarker
  if (trackingConfig.doAcquireHandLandmarks){
    myHandLandmarker = await HandLandmarker.createFromOptions(vision, {
      numHands: trackingConfig.maxNumHands,
      runningMode: "VIDEO",
      baseOptions: {
        delegate: trackingConfig.cpuOrGpuString,
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
    });
  }

  //Gesture Recognition
  //https://codepen.io/mediapipe-preview/pen/zYamdVd
  //https://mediapipe-studio.webapps.google.com/studio/demo/gesture_recognizer
  if(trackingConfig.doAcquireHandGesture){
    myGestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        delegate: trackingConfig.cpuOrGpuString,
        modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
      },
      runningMode:  "VIDEO",
      numHands: trackingConfig.maxNumHands
  });

  }

  
  // Pose (Body) Landmark Tracking: 
  // https://codepen.io/mediapipe-preview/pen/abRLMxN
  // https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
  if (trackingConfig.doAcquirePoseLandmarks){
    const poseModelLite = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
    const poseModelFull = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
    let poseModel = poseModelLite;
    poseModel = (trackingConfig.poseModelLiteOrFull == "full") ? poseModelFull:poseModelLite;
    myPoseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      numPoses: trackingConfig.maxNumPoses,
      runningMode: "VIDEO",
      baseOptions: {
        modelAssetPath: poseModel,
        delegate:trackingConfig.cpuOrGpuString,
      },
    });
  }
  
  // Face Landmark Tracking:
  // https://codepen.io/mediapipe-preview/pen/OJBVQJm
  // https://developers.google.com/mediapipe/solutions/vision/face_landmarker
  if (trackingConfig.doAcquireFaceLandmarks){
    myFaceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      numFaces: trackingConfig.maxNumFaces,
      runningMode: "VIDEO",
      outputFaceBlendshapes:trackingConfig.doAcquireFaceMetrics,
      baseOptions: {
        delegate: trackingConfig.cpuOrGpuString,
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      },
    });
  }
}

//------------------------------------------
async function predictWebcam() {
  let startTimeMs = performance.now();
  if (lastVideoTime !== myCapture.elt.currentTime) {
    if (trackingConfig.doAcquireHandLandmarks && myHandLandmarker) {
      handLandmarks = myHandLandmarker.detectForVideo(myCapture.elt,startTimeMs);
    }
    if (trackingConfig.doAcquireHandGesture && myGestureRecognizer) {
      gestureResults = myGestureRecognizer.recognizeForVideo(myCapture.elt,startTimeMs);
    }
    if (trackingConfig.doAcquirePoseLandmarks && myPoseLandmarker) {
      poseLandmarks = myPoseLandmarker.detectForVideo(myCapture.elt,startTimeMs);
    }
    if (trackingConfig.doAcquireFaceLandmarks && myFaceLandmarker) {
      faceLandmarks = myFaceLandmarker.detectForVideo(myCapture.elt,startTimeMs);
    }
    lastVideoTime = myCapture.elt.currentTime;
  }
  window.requestAnimationFrame(predictWebcam);//It's running in it's own cycle
}


//------------------------------------------
function setup() {
  createCanvas(640, 480);
  myCapture = createCapture(VIDEO);
  myCapture.size(320, 240);
  myCapture.hide();
  predictWebcam();//ONLY CALL ONCE at start
}

function draw() {
  background("white");
  drawVideoBackground();
  drawHandPoints();
  drawGestures();
  drawPosePoints(); 
  drawFacePoints();
  drawFaceMetrics();
  drawDiagnosticInfo();
}


//------------------------------------------
function drawVideoBackground() {
  push();
  //Flip the camera here to make it like a mirror
  translate(width, 0);
  scale(-1, 1);
  tint(255, 255, 255, 72);
  image(myCapture, 0, 0, width, height);
  tint(255);
  pop();
}

//------------------------------------------
// HANDS: 21 2D landmarks per hand, up to maxNumHands at once
function drawHandPoints() {
  if (trackingConfig.doAcquireHandLandmarks) {
    if (handLandmarks && handLandmarks.landmarks) {
      drawHandLandMarks(handLandmarks.landmarks);
    }
  }
}

//Re-usable function for drawing hand lanmarks
function drawHandLandMarks(landmarks) {
  const nHands = landmarks.length;
  if (nHands > 0) {
    // Draw lines connecting the joints of the fingers
    noFill();
    stroke("black");
    
    strokeWeight(2.0);
    for (let h = 0; h < nHands; h++) {
      let joints = landmarks[h];
      drawConnectors(joints, HANDLANDMARKER_PALM);
      drawConnectors(joints, HANDLANDMARKER_THUMB);
      drawConnectors(joints, HANDLANDMARKER_INDEX_FINGER);
      drawConnectors(joints, HANDLANDMARKER_MIDDLE_FINGER);
      drawConnectors(joints, HANDLANDMARKER_RING_FINGER);
      drawConnectors(joints, HANDLANDMARKER_PINKY);
    }
    
    // Draw just the joints of the hands
    strokeWeight(1.0);
    stroke("black");
    fill("red");
    for (let h = 0; h < nHands; h++) {
      let joints = landmarks[h];
      for (let i = 0; i <= 20; i++) {
        let px = joints[i].x;
        let py = joints[i].y;
        px = map(px, 0, 1, width, 0);
        py = map(py, 0, 1, 0, height);
        circle(px, py, 9);
      }
    }
  }
}


//------------------------------------------
function drawGestures(){
  if (trackingConfig.doAcquireHandGesture) {

    //draw Gesture Hand Landmarks
    if (gestureResults && gestureResults.landmarks) {
      drawHandLandMarks(gestureResults.landmarks);
    }

    //Handle Gesture Detection
    if(gestureResults && gestureResults.gestures){
      for (let i = 0; i < gestureResults.gestures.length; i++) {
        noStroke();
        fill(255, 0, 0);
        textSize(20);
        let name = gestureResults.gestures[i][0].categoryName;
        let score = gestureResults.gestures[i][0].score;
        let right_or_left = gestureResults.handednesses[i][0].hand;
        let pos = {
          x: gestureResults.landmarks[i][0].x * width,
          y: gestureResults.landmarks[i][0].y * height,
        };
        textSize(48);
        fill(0);
        textAlign(CENTER, CENTER);
        let mapX = map(pos.x, 0, width, width, 0);//because we have flipped the camera
        console.log(pos.x,mapX);
        text(name, mapX, pos.y);
      }
    }

  }
}


//------------------------------------------
// 33 joints of the body. See landmarks.js for the list. 
function drawPosePoints(){
  if (trackingConfig.doAcquirePoseLandmarks) {
    if (poseLandmarks && poseLandmarks.landmarks) {
      const nPoses = poseLandmarks.landmarks.length;
      if (nPoses > 0) {
        
        // Draw lines connecting the joints of the body
        noFill();
        stroke("darkblue");
        strokeWeight(2.0);
        for (let h = 0; h < nPoses; h++) {
          let joints = poseLandmarks.landmarks[h];
          drawConnectors(joints, PoseLandmarker.POSE_CONNECTIONS);
        }
      }
    }
  }
}


//------------------------------------------
// Tracks 478 points on the face. 
function drawFacePoints() {
  if (trackingConfig.doAcquireFaceLandmarks) {
    if (faceLandmarks && faceLandmarks.faceLandmarks) {
      const nFaces = faceLandmarks.faceLandmarks.length;
      if (nFaces > 0) {
        for (let f = 0; f < nFaces; f++) {
          let aFace = faceLandmarks.faceLandmarks[f];
          if (aFace) {
            let nFaceLandmarks = aFace.length;
            
            noFill();
            stroke("black");
            strokeWeight(1.0);
            for (let i = 0; i < nFaceLandmarks; i++) {
              let px = aFace[i].x;
              let py = aFace[i].y;
              px = map(px, 0, 1, width, 0);
              py = map(py, 0, 1, 0, height);
              circle(px, py, 1);
            }
            
            noFill();
            stroke("black");
            strokeWeight(2.0);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_LIPS);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS);
            drawConnectors(aFace, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS);
            drawConnectors(aFace, FACELANDMARKER_NOSE); // Google offers no nose
          }
        }
      }
    }
  }
}

function drawFaceMetrics(){
  if (trackingConfig.doAcquireFaceLandmarks && 
      trackingConfig.doAcquireFaceMetrics){
    if (faceLandmarks && faceLandmarks.faceBlendshapes) {
      const nFaces = faceLandmarks.faceLandmarks.length;
      for (let f = 0; f < nFaces; f++) {
        let aFaceMetrics = faceLandmarks.faceBlendshapes[f];
        if (aFaceMetrics){
          
          fill('black'); 
          textSize(7); 
          let tx = 40; 
          let ty = 40; 
          let dy = 8.5;
          let vx0 = tx-5; 
          let vx1 = tx-35;
          
          let nMetrics = aFaceMetrics.categories.length; 
          for (let i=1; i<nMetrics; i++){
            let metricName = aFaceMetrics.categories[i].categoryName;
            noStroke();
            text(metricName, tx,ty); 
            
            let metricValue = aFaceMetrics.categories[i].score;
            let vx = map(metricValue,0,1,vx0,vx1);
            stroke(0,0,0); 
            strokeWeight(2.0); 
            line(vx0,ty-2, vx,ty-2); 
            stroke(0,0,0,20);
            line(vx0,ty-2, vx1,ty-2); 
            ty+=dy;
          }
        }
      }
    }
  }
}


//------------------------------------------
function drawConnectors(landmarks, connectorSet) {
  if (landmarks) {
    let nConnectors = connectorSet.length;
    for (let i=0; i<nConnectors; i++){
      let index0 = connectorSet[i].start; 
      let index1 = connectorSet[i].end;
      let x0 = map(landmarks[index0].x, 0,1, width,0);
      let y0 = map(landmarks[index0].y, 0,1, 0,height);
      let x1 = map(landmarks[index1].x, 0,1, width,0);
      let y1 = map(landmarks[index1].y, 0,1, 0,height);
      line(x0,y0, x1,y1); 
    }
  }
}

//------------------------------------------
function drawDiagnosticInfo() {
  noStroke();
  fill("black");
  textSize(12); 
  text("FPS: " + int(frameRate()), 40, 30);
}
