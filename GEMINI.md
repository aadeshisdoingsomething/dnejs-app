# Dynamic Neural Engine - Context & Guidelines

This document provides architectural overview, development conventions, and technical context for the Dynamic Neural Engine project.

## Project Overview
The **Dynamic Neural Engine** is a pure JavaScript implementation of a feed-forward neural network designed for browser-based Natural Language Processing (NLP). It operates entirely client-side without external dependencies or heavy frameworks, using a custom-built "brain" that learns patterns between user queries and AI responses in real-time.

### Key Technologies
- **Frontend**: HTML5, Tailwind CSS (via CDN), Lucide Icons.
- **Logic**: Vanilla JavaScript (ES Modules).
- **Persistence**: Browser `localStorage`.
- **Architecture**: Modular functional-style JavaScript.

## Architecture & Module Map

The codebase is organized into specialized modules:

- **`index.html`**: Entry point and UI structure.
- **`js/main.js`**: Orchestrator that initializes the app, handles high-level events, and connects UI to logic.
- **`js/model.js`**: The core neural network logic (`AdvancedBrain` class). Implements:
    - Feed-forward execution with ReLU (hidden layer) and Softmax (output layer).
    - True Mini-Batch Backpropagation (accumulating and applying gradients).
    - Flat 1D `Float32Array` memory structures for maximum V8 engine performance.
    - Dynamic weight matrix expansion to accommodate growing vocabulary.
- **`js/worker.js`**: Dedicated background Web Worker that runs the computationally heavy `AdvancedBrain` training loop.
- **`js/state.js`**: Centralized global state object, including the vocabulary, corpus, and brain instance.
- **`js/nlp.js`**: Text processing utilities:
    - Tokenization and formatting utilities.
    - **Auto-Regressive Pipeline**: Translates text into sequential context encodings, enabling the model to predict next-word probability distributions instead of block bag-of-words outputs.
- **`js/training.js`**: Manages the message bridging between the UI thread and `worker.js`, handling batched array transfers.
- **`js/storage.js`**: Handles saving and loading the model state (weights, vocabulary, corpus) to/from `localStorage`.
- **`js/ui.js`**: DOM element references and rendering functions (chat messages, corpus list, and matrix visualizer).

## Building and Running

Since this is a client-side application with no build step, it can be run directly.

- **Direct Execution**: Open `index.html` in any modern web browser.
- **Local Development**: To avoid potential CORS issues and for a better development experience, serve it using a local server:
  ```bash
  npx serve .
  ```

## Development Conventions

- **State Management**: Always use the central `state` object in `js/state.js`. Avoid local state for data that needs to persist or be shared across modules.
- **Persistence**: Updates to the brain (weights) or corpus should be followed by a call to `saveData()` from `js/storage.js`.
- **NLP Pipeline**: When adding features, ensure they integrate with the `encode`/`decode` flow in `js/nlp.js`.
- **Visuals**: Styling is handled via Tailwind CSS classes in `index.html` and `js/ui.js`.
- **Performance**: Training is computationally intensive. The loop in `js/training.js` acts as an orchestrator, passing data to a dedicated Web Worker (`js/worker.js`) to guarantee a smooth, unblocked UI.

## Current Limitations & Future Work
- **Network Depth**: The model currently uses a single hidden layer (shallow network); future tasks include transitioning to multiple hidden layers.
- **Vocabulary Constraints**: Vocabulary is strictly additive; removing bad nodes requires a full "Core Nuke" (clearing `localStorage`).
- **Context Drift**: As auto-regressive generation length increases, a sliding context window may need to be strictly enforced to prevent grammatical drifting or infinite loops.

*(Legacy Note: Early versions utilized a Bag-of-Words approach with global Sigmoid activations and were executed on the main UI thread. Current docs reflect the high-performance V2 engine.)*
