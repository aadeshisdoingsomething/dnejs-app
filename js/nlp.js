import { state } from './state.js';

export function tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 0);
}

export function encode(text) {
    return encodeSequence(tokenize(text));
}

export function encodeSequence(wordsArray) {
    // 1. Convert sequence of words to integer indexes from vocab
    let sequenceIndices = wordsArray.map(word => {
        let idx = state.vocab.indexOf(word);
        // If word is unknown, fallback to index 0 (<PAD>)
        return idx !== -1 ? idx : 0;
    });

    // 2. The Embedding Layer in model.js demands a strict, fixed-length array (contextWindowSize)
    let paddedSequence = new Array(state.contextWindowSize).fill(0); // Fill with <PAD> index 0

    // 3. Right-align the words into the fixed window so the model reads normally left-to-right
    // E.g. [PAD, PAD, PAD, "how", "are", "you"]
    let startIdx = state.contextWindowSize - sequenceIndices.length;
    for (let i = 0; i < sequenceIndices.length; i++) {
        if (startIdx + i >= 0) {
            paddedSequence[startIdx + i] = sequenceIndices[i];
        }
    }

    // Return the pure integer array. The Embedding layer natively handles semantic geometry.
    return paddedSequence;
}

export function updateEncodedCorpus() {
    state.encodedCorpus = [];
    state.corpus.forEach(c => {
        const qWords = tokenize(c.q);
        const aWords = tokenize(c.a);

        // Sequence Builder: [Question] -> <SOS> -> [Answer] -> <EOS>
        // Start context with the Question, plus the Start-Of-Sequence trigger token
        let currentContext = [...qWords, "<SOS>"];

        // The very first word the model must predict is the FIRST word of the answer, 
        // based on the context of the question + <SOS>.
        // Then it Auto-Regressively slides through predicting each subsequent answer word.
        for (let i = 0; i < aWords.length; i++) {
            const targetWord = aWords[i];
            const targetVec = new Array(state.vocab.length).fill(0);
            const targetIdx = state.vocab.indexOf(targetWord);
            if (targetIdx !== -1) targetVec[targetIdx] = 1;

            const contextSlice = currentContext.slice(-state.contextWindowSize);

            state.encodedCorpus.push({
                in: encodeSequence(contextSlice),
                out: targetVec
            });

            // Auto-Regressive Shift: append target to context for next prediction
            currentContext.push(targetWord);
        }

        // Finally, teach the model to predictably output <EOS> token when finished
        const eosVec = new Array(state.vocab.length).fill(0);
        eosVec[state.vocab.indexOf('<EOS>')] = 1;
        const finalContextSlice = currentContext.slice(-state.contextWindowSize);
        state.encodedCorpus.push({
            in: encodeSequence(finalContextSlice),
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
