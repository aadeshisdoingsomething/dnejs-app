# Dynamic Neural Engine

A pure JavaScript implementation of a neural network for Natural Language Processing, running entirely in your browser. No external APIs, no heavy frameworks—just raw math and logic.

### 🧠 What is it?
This project is a functional, browser-based "brain" that you can train in real-time. It uses a custom-built neural network to learn patterns between user queries and AI responses. You can teach it manually or import a large corpus of data to see it evolve.

### ⚙️ How it works (Current Architecture)
- **Neural Network Architecture**: A feed-forward network currently using one hidden layer with **ReLU** activations and **Softmax** output, optimized with **Cross-Entropy loss**. It uses true mini-batch gradient descent (accumulated gradients) with momentum.
- **Dynamic Vocabulary**: Automatically expands 1D `Float32Array` matrices on the fly as it encounters new words.
- **NLP Pipeline**: Employs an **Auto-Regressive** generation system. It encodes input as sequential word context, predicting the *next word* in the sequence until hitting an end-of-sentence token.
- **Web Worker Training**: Heavy matrix multiplication and backpropagation are fully offloaded to a background `Worker` thread, guaranteeing the UI never drops frames.
- **Visualizer**: Includes a live matrix visualizer that lets you see the dynamic weight vectors changing as the model trains.
- **Storage**: All training data and model states are persisted in your browser's `localStorage`.

*Legacy Architecture (For Historical Reference): Originally built using a Bag-of-Words encoder, Sigmoid activations everywhere, and a synchronous `setTimeout` training loop on the main thread.*

### 🚀 How to run
Since this is a client-side application, running it is simple:
1.  Open `index.html` directly in any modern web browser.
2.  *Alternatively*, serve it using a local server for a better experience:
    ```bash
    npx serve .
    ```
3.  Once open, use the **Manual** tab to add a few Q&A pairs, then hit **Train Until Target** to see the model learn your inputs. 
4.  Alternatively, you can use the **Corpus** tab to import a large corpus of data to see it evolve.

### 🔮 Future Goals
- **Make it Smarter**: Move beyond a simple feed-forward architecture to include multiple hidden layers or implement Transformer-style attention mechanisms for better context handling.
- **Exportable Models**: Add functionality to export the trained weights and vocabulary as a JSON file, allowing you to share or deploy your "trained brain" elsewhere.
- **Performance Optimization**: Utilize Web Workers or WebGL/WebGPU to speed up training for larger datasets.
