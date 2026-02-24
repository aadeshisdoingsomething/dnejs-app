import { state } from './state.js';
import { encode } from './nlp.js';
import { saveData } from './storage.js';
import { DOM } from './ui.js';

export function trainBatch() {
    if (!state.isTraining) return;

    const target = parseFloat(DOM.uiTargetLoss.value) || 0;
    let totalError = 0;
    const pairs = state.corpus.map(c => ({ in: encode(c.q), out: encode(c.a) }));

    // Batch size 100
    for (let k = 0; k < 100; k++) {
        const sample = pairs[Math.floor(Math.random() * pairs.length)];
        totalError += state.brain.train(sample.in, sample.out);
    }

    state.currentLoss = totalError / 100;
    state.iterations += 100;

    DOM.uiLoss.innerText = state.currentLoss.toFixed(5);
    DOM.trainText.innerText = `TRAINING (${state.iterations})`;

    if (state.currentLoss <= target) {
        toggleTraining(true); // Force stop
        saveData(); // Save weights once target is reached
    } else {
        state.trainingTimeout = setTimeout(trainBatch, 0); // Asynchronous infinite loop
    }
}

export function toggleTraining(forceStop = false) {
    if (state.isTraining || forceStop) {
        state.isTraining = false;
        clearTimeout(state.trainingTimeout);
        DOM.btnTrain.className = "w-full py-4 rounded-2xl font-black text-sm flex items-center justify-center gap-2 transition-all bg-indigo-600 text-white hover:bg-indigo-500";
        DOM.trainIcon.setAttribute('data-lucide', 'play');
        DOM.trainText.innerText = 'TRAIN UNTIL TARGET';
    } else {
        state.isTraining = true;
        DOM.btnTrain.className = "w-full py-4 rounded-2xl font-black text-sm flex items-center justify-center gap-2 transition-all bg-rose-500/20 text-rose-500 border border-rose-500/30";
        DOM.trainIcon.setAttribute('data-lucide', 'activity');
        DOM.trainIcon.classList.add('animate-pulse');
        trainBatch();
    }
    if (window.lucide) window.lucide.createIcons();
}
