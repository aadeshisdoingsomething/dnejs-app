# Dynamic Neural Engine - Improvement Roadmap

This document outlines specific, actionable steps to improve the efficiency, speed, and "intelligence" of the current neural network architecture without relying on external libraries.

## 1. [x] Optimize the Training Loop (Immediate Efficiency Gain)
**Current Issue:** In `js/training.js`, the `trainBatch` function re-encodes the *entire* corpus every single time it runs (which is every frame due to `setTimeout`).
```javascript
// This runs on every iteration and is very expensive!
const pairs = state.corpus.map(c => ({ in: encode(c.q), out: encode(c.a) }));
```
**Action:**
- Cache the encoded pairs in `state.js` whenever a new item is added to the corpus.
- Modify `trainBatch` to use the pre-encoded cached pairs instead of mapping the corpus continuously. This will drastically reduce CPU load and speed up iterations.

## 2. [x] Implement True Mini-Batch Gradient Descent
**Current Issue:** The `trainBatch` loops 100 times and calls `state.brain.train()` for each random sample. Inside `train()`, the weights are updated immediately. This is actually Stochastic Gradient Descent (SGD) run 100 times, which is noisy and computationally expensive due to constant array mutations.
**Action:**
- Modify the `train` method in `js/model.js` to separate gradient calculation from weight updating.
- Create an `accumulateGradients(inputVector, targetVector)` method.
- Create an `applyGradients()` method.
- In `trainBatch`, loop 100 times calling `accumulateGradients`, and then call `applyGradients` once at the end of the batch. This will vectorize better and provide a more stable learning path.

## 3. [x] Move from Bag-of-Words to Auto-Regressive Generation (Make it Smarter)
**Current Issue:** `js/nlp.js` uses a Bag-of-Words (BoW) encoding. It loses all word order. "The dog bit the man" and "The man bit the dog" have the exact same input vector. Furthermore, `decode()` just outputs the top 3 words based on their index in the vocabulary, meaning the output grammar will always be broken.
**Action:**
- **Input:** Implement N-grams (e.g., pairs of words) or positional encoding so the input vector retains sequential context.
- **Output:** Change the model to predict the *next word* rather than the whole sentence at once. 
- **Inference:** Instead of a single pass, feed the input to get the first word, then append that word to the input and feed it again to get the second word, stopping when an `<EOS>` (End of Sentence) token is predicted.

## 4. [x] Upgrade Activations & Loss Functions
**Current Issue:** The network uses Sigmoid activations everywhere and calculates loss by simply subtracting vectors (`target - output`).
**Action:**
- **Hidden Layer:** Replace `sigmoid` with `ReLU` (`Math.max(0, x)`) in `js/model.js`. It's computationally cheaper (no `Math.exp`) and prevents the vanishing gradient problem, allowing for faster convergence.
- **Output Layer:** If shifting to predicting one word at a time, change the output activation to `Softmax` to generate a true probability distribution across the vocabulary.
- **Loss Calculation:** Implement Cross-Entropy Loss instead of the current Mean Absolute Error approximation. Cross-Entropy heavily penalizes the model for being confidently wrong, which is standard for classification/NLP.

## 5. [x] Offload Computation to Web Workers
**Current Issue:** Matrix multiplication is happening on the main browser thread via nested `for` loops. While `setTimeout` prevents complete freezing, the UI will still stutter during heavy training.
**Action:**
- Move `model.js` and `training.js` execution into a standard JavaScript Web Worker. 
- The main thread should only handle DOM updates and pass messages (like new corpus data or "start training" commands) to the worker. The worker will send back the current loss and iteration count.

## 6. [x] Flatten Matrices to Typed Arrays (Performance)
**Current Issue:** Weights and biases are stored as Arrays of Arrays (e.g., `Array.from({ length: hiddenNodes }, () => ...)`). V8 engine struggles to optimize nested dynamic arrays.
**Action:**
- Refactor `weightsIH`, `weightsHO`, etc., to use flat 1D `Float32Array` objects. 
- Address them using math: `index = row * width + col`. Typed arrays provide a massive speed boost in JavaScript for heavy numerical computation.

## 7. [x] Implement Simple Positional Encoding
Current Issue: The model receives a sequence of tokens, but the weights for a specific word (e.g., "is") are the same whether it appears at the start, middle, or end of a sentence. This makes it hard for the model to learn that "What" usually starts a sentence and "<EOS>" ends one.
Action:

The Concept: Augment the input vector with a "time" signal.

Implementation: When calling predict(), don't just send the one-hot encoded word vector. Append a normalized value (e.g., index / maxTokens) as an additional input node, or add a small unique value to the word's activation based on its position in the string.

Architectural Change: Increase this.vocabSize by 1 in the constructor to account for this "Position Node." This gives the weightsIH a specific "temporal" weight to learn from, allowing the model to distinguish between "Is math..." and "...math is."

8. Implement a Sliding Context Window
Current Issue: As the AI generates longer sentences, the "noise" from the beginning of the sentence can confuse the next-word prediction, leading to infinite loops or "humans humans" spam.
Action:

The Concept: Limit the "Memory" of the model during inference.

Implementation: In main.js, when performing the generation loop, instead of passing the entire currentContext to the encoder, only pass the last N tokens (e.g., currentContext.slice(-3)).

Benefit: This mimics how humans focus on the immediate phrase structure. It keeps the input vector size constant while ensuring the model stays "on track" with the most recent grammatical context.

Deepening the Brain (Task #9)

The Goal: Add a second hidden layer.

Why: Your current architecture is Input -> Hidden -> Output. This is a "shallow" network. Adding one more layer (Hidden1 -> Hidden2) allows the model to learn hierarchical features.

Layer 1 learns "What words go together."

Layer 2 learns "What do these words mean together."

Difficulty: High (7/10). You have to pass the error backwards through two sets of weights. But since you have Mini-Batching and Typed Arrays done, the math will stay stable.