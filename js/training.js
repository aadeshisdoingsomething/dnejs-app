import { state } from './state.js';
import { encode } from './nlp.js';
import { saveData } from './storage.js';
import { DOM } from './ui.js';

let trainingWorker = null;

function initWorker() {
    if (trainingWorker) return;
    trainingWorker = new Worker('./js/worker.js', { type: 'module' });

    trainingWorker.onmessage = function (e) {
        if (!state.isTraining) return;

        const { type, payload } = e.data;

        if (type === 'PROGRESS') {
            state.currentLoss = payload.currentLoss;
            state.iterations = payload.iterations;
            state.brain.setRawState(payload.rawState); // Kept in sync for UI/Visualizer

            DOM.uiLoss.innerText = state.currentLoss.toFixed(5);
            DOM.trainText.innerText = `TRAINING (${state.iterations})`;
        } else if (type === 'DONE') {
            toggleTraining(true);
            saveData();
        }
    };
}

export function syncWorkerData() {
    if (!trainingWorker) return;

    const data = {
        type: 'UPDATE_CORPUS',
        payload: {
            vocabSize: state.vocab.length,
            encodedCorpus: state.encodedCorpus && state.encodedCorpus.length > 0
                ? state.encodedCorpus
                : state.corpus.map(c => ({ in: encode(c.q), out: encode(c.a) }))
        }
    };
    trainingWorker.postMessage(data);
}

export function toggleTraining(forceStop = false) {
    if (!trainingWorker) initWorker();

    if (state.isTraining || forceStop) {
        state.isTraining = false;
        trainingWorker.postMessage({ type: 'STOP' });

        DOM.btnTrain.className = "w-full py-4 rounded-2xl font-black text-sm flex items-center justify-center gap-2 transition-all bg-indigo-600 text-white hover:bg-indigo-500";
        DOM.trainIcon.setAttribute('data-lucide', 'play');
        DOM.trainText.innerText = 'TRAIN UNTIL TARGET';
    } else {
        state.isTraining = true;

        const target = parseFloat(DOM.uiTargetLoss.value) || 0;

        trainingWorker.postMessage({
            type: 'INIT',
            payload: {
                vocabSize: state.vocab.length,
                hiddenNodes: state.brain.hiddenNodes,
                rawState: state.brain.getRawState(),
                encodedCorpus: state.encodedCorpus && state.encodedCorpus.length > 0
                    ? state.encodedCorpus
                    : state.corpus.map(c => ({ in: encode(c.q), out: encode(c.a) })),
                targetLoss: target,
                iterations: state.iterations
            }
        });

        trainingWorker.postMessage({ type: 'START' });

        DOM.btnTrain.className = "w-full py-4 rounded-2xl font-black text-sm flex items-center justify-center gap-2 transition-all bg-rose-500/20 text-rose-500 border border-rose-500/30";
        DOM.trainIcon.setAttribute('data-lucide', 'activity');
        DOM.trainIcon.classList.add('animate-pulse');
    }
    if (window.lucide) window.lucide.createIcons();
}
