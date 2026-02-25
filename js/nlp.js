import { state } from './state.js';

export function tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 0);
}

export function encode(text) {
    return encodeSequence(tokenize(text));
}

export function encodeSequence(wordsArray) {
    const vec = new Array(state.vocab.length).fill(0);
    // Continuous Bag Of Words implementation (Sums the input context together)
    wordsArray.forEach(w => {
        const idx = state.vocab.indexOf(w);
        if (idx !== -1) vec[idx] += 1; // Used additive logic over fixed `1` to represent frequency
    });
    return vec;
}

export function updateEncodedCorpus() {
    state.encodedCorpus = [];
    state.corpus.forEach(c => {
        const qWords = tokenize(c.q);
        const aWords = tokenize(c.a);

        // Base context starts with just the question
        let currentContext = [...qWords];

        // For each answer word, predict it based on current context
        for (let i = 0; i < aWords.length; i++) {
            const targetWord = aWords[i];
            const targetVec = new Array(state.vocab.length).fill(0);
            const targetIdx = state.vocab.indexOf(targetWord);
            if (targetIdx !== -1) targetVec[targetIdx] = 1;

            state.encodedCorpus.push({
                in: encodeSequence(currentContext),
                out: targetVec
            });

            // Auto-Regressive Shift: append target to context for next prediction
            currentContext.push(targetWord);
        }

        // Finally, teach the model to predictably output <EOS> token when finished
        const eosVec = new Array(state.vocab.length).fill(0);
        eosVec[state.vocab.indexOf('<EOS>')] = 1;
        state.encodedCorpus.push({
            in: encodeSequence(currentContext),
            out: eosVec
        });
    });
}

export function decode(vector) {
    let maxIdx = 0;
    let maxVal = -Infinity;

    for (let i = 0; i < vector.length; i++) {
        if (vector[i] > maxVal) {
            maxVal = vector[i];
            maxIdx = i;
        }
    }

    return state.vocab[maxIdx] || "<PAD>";
}
