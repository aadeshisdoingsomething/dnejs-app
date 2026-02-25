import { AdvancedBrain } from './model.js';

export const state = {
    vocab: [
        "<PAD>", "<EOS>", "hi", "hello", "i", "am", "robot", "ai", "bot",
        "good", "bad", "yes", "no", "what", "is", "your", "name",
    ],
    corpus: [
        { q: "hi", a: "hello" },
    ],
    encodedCorpus: [],
    brain: null,
    messages: [],
    isTraining: false,
    trainingTimeout: null,
    currentLoss: 50,
    iterations: 0
};

// Initialize the brain
state.brain = new AdvancedBrain(state.vocab.length, 32);
