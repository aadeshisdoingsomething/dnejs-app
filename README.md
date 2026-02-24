# Dynamic Neural Engine

A pure JavaScript implementation of a neural network for Natural Language Processing, running entirely in your browser. No external APIs, no heavy frameworks—just raw math and logic.

### 🧠 What is it?
This project is a functional, browser-based "brain" that you can train in real-time. It uses a custom-built neural network to learn patterns between user queries and AI responses. You can teach it manually or import a large corpus of data to see it evolve.

### ⚙️ How it works
- **Neural Network Architecture**: A feed-forward network with one hidden layer, utilizing sigmoid activation functions and backpropagation with momentum for learning.
- **Dynamic Vocabulary**: As you teach the model new words, it automatically expands its input and output matrices to accommodate the growing vocabulary.
- **NLP Pipeline**: It tokenizes text, encodes it into vectors for the network to process, and decodes the output vectors back into human-readable text.
- **Visualizer**: Includes a live matrix visualizer that lets you see the synaptic weights changing as the model trains.
- **Storage**: All training data and model states are persisted in your browser's `localStorage`.

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
