import { AdvancedBrain } from './model.js';
import { encode } from './nlp.js';

let brain = null;
let encodedCorpus = [];
let targetLoss = 0.005;
let isTraining = false;
let iterations = 0;
let currentLoss = 50;
let trainingTimeout = null;

// Worker message handler
self.onmessage = function (e) {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT':
            // Instantiate brain and restore live weights
            brain = new AdvancedBrain(payload.vocabSize, payload.hiddenNodes);
            brain.setRawState(payload.rawState);
            encodedCorpus = payload.encodedCorpus;
            targetLoss = payload.targetLoss;
            iterations = payload.iterations || 0;
            break;

        case 'START':
            if (!isTraining && brain && encodedCorpus.length > 0) {
                isTraining = true;
                trainLoop();
            }
            break;

        case 'STOP':
            isTraining = false;
            if (trainingTimeout) clearTimeout(trainingTimeout);
            break;

        case 'UPDATE_CORPUS':
            // User added a new word or sentence
            encodedCorpus = payload.encodedCorpus;
            if (brain && payload.vocabSize > brain.vocabSize) {
                brain.expandVocab(payload.vocabSize);
            }
            break;

        case 'SET_TARGET':
            targetLoss = payload.targetLoss;
            break;
    }
};

async function trainLoop() {
    if (!isTraining || !brain || encodedCorpus.length === 0) return;

    const batchSize = 100;

    // trainBatch now handles array sampling, tensor creation, and fitting
    currentLoss = await brain.trainBatch(encodedCorpus, batchSize);
    iterations += batchSize;

    if (iterations % 1000 === 0) { // Every 10 loops of 100 batch size
        brain.decayLearningRate();
    }

    // Send progress back to main thread
    self.postMessage({
        type: 'PROGRESS',
        payload: {
            currentLoss,
            iterations,
            rawState: brain.getRawState()
        }
    });

    if (currentLoss <= targetLoss) {
        isTraining = false;
        self.postMessage({ type: 'DONE' });
    } else {
        // Yield to allow message processing, then loop
        trainingTimeout = setTimeout(trainLoop, 10);
    }
}
