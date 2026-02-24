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
    - Feed-forward execution.
    - Backpropagation with momentum.
    - Dynamic weight matrix expansion to accommodate growing vocabulary.
    - Serialization/Deserialization for persistence.
- **`js/state.js`**: Centralized global state object, including the vocabulary, corpus, and brain instance.
- **`js/nlp.js`**: Text processing utilities:
    - `tokenize`: Cleans and splits text.
    - `encode`: Converts text into multi-hot vectors for the network.
    - `decode`: Converts network output vectors back into words using a probability threshold.
- **`js/training.js`**: Manages the training loop, loss calculation, and asynchronous execution using `setTimeout`.
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
- **Performance**: Training is computationally intensive; the loop in `js/training.js` is designed to be asynchronous to prevent UI freezing. Keep batch sizes and iterations balanced.

## Current Limitations & Future Work
- The model currently uses a single hidden layer and sigmoid activation.
- Vocabulary is currently additive; removing words requires a "Core Nuke" (clearing `localStorage`).
- Inference uses a simple probability threshold for decoding, which can lead to disjointed responses if the model is under-trained.
