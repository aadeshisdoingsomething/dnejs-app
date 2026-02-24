import { state } from './state.js';

export function tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 0);
}

export function encode(text) {
    const vec = new Array(state.vocab.length).fill(0);
    tokenize(text).forEach(w => {
        const idx = state.vocab.indexOf(w);
        if (idx !== -1) vec[idx] = 1;
    });
    return vec;
}

export function decode(vector) {
    // Lower threshold to see more words
    const threshold = 0.15;

    return vector
        .map((val, idx) => ({ val, word: state.vocab[idx] }))
        // Keep words the model is even slightly sure about
        .filter(item => item.val > threshold)
        // REMOVED .sort() - we want to keep the order the model picked!
        .slice(0, 3)
        .map(item => item.word)
        .join(" ") || "... (shrugs)";
}
