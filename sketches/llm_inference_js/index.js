// Modified from here:
// https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/llm_inference/js
// See also this:
// https://chatgpt.com/share/69779cba-222c-8005-aeb2-d319bb36d81c

// ---------------------------------------------------------------------------------------- //

// Copyright 2024 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ---------------------------------------------------------------------------------------- //

import { FilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai';
// Handle markdown
import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";

/* Update the file name */
const modelFileName = 'gemma3-1b-it-int4.task';

const input = document.getElementById('input');
const output = document.getElementById('output');
const submit = document.getElementById('submit');

// ctrl+enter to send message
input.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") {
    if (input.value.trim() !== "" && !submit.disabled) {
      e.preventDefault(); // avoid newline
      submit.click();
    }
  }
});

/**
 * Display newly generated partial results to the output text box.
 */
function displayPartialResults(partialResults, complete) {
  input.value = "";
  const lastP = output.lastChild;

  // store raw markdown on the element
  lastP._md = (lastP._md || "") + partialResults;

  // render markdown → html
  lastP.innerHTML = marked.parse(lastP._md);

  if (complete) {
    if (!lastP.textContent.trim()) {
      lastP.textContent = "(No reply...)";
    }
    submit.disabled = false;
  }
}

/**
 * Main function to run LLM Inference.
 */
async function runDemo() {
  const genaiFileset = await FilesetResolver.forGenAiTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm');
  let llmInference;

  submit.onclick = () => {

    input.value = input.value.trim();

    console.log(input.value);
    // don't process empty
    if (input.value.length == 0) return;

    const p1 = document.createElement("p");
    p1.innerText = input.value;
    p1.classList += "bubble right";
    output.appendChild(p1);

    const p2 = document.createElement("p");
    p2.classList += "bubble left";
    output.appendChild(p2);
    submit.disabled = true;

    llmInference.generateResponse(input.value, displayPartialResults);

  };


  submit.value = 'Loading model...'
  LlmInference
      .createFromOptions(genaiFileset, {
        baseOptions: {modelAssetPath: modelFileName},
        maxTokens: 1280,      // The maximum number of tokens (input tokens + output
                              // tokens) the model handles.
        // randomSeed: 1,     // The random seed used during text generation.
        // topK: 40,          // The number of tokens the model considers at each step of
        //                    // generation. Limits predictions to the top k most-probable
        //                    // tokens. Setting randomSeed is required for this to make
        //                    // effects.
        // temperature: 1.0,  // The amount of randomness introduced during generation.
        //                    // Setting randomSeed is required for this to make effects.
      })
      .then(llm => {
        llmInference = llm;
        submit.disabled = false;
        submit.value = 'Send'
      })
      .catch((e) => {
        console.log(e);
        alert('Failed to initialize the task. Are you in Chrome?');
      });
}

runDemo();
