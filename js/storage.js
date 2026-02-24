import { state } from './state.js';
import { DOM, renderCorpus } from './ui.js';

export function loadData() {
    try {
        const dataStr = localStorage.getItem('neuralEngineData');
        if (dataStr) {
            const data = JSON.parse(dataStr);
            if (data.vocab) state.vocab = data.vocab;
            if (data.corpus) state.corpus = data.corpus;
            if (data.brainState && state.brain) state.brain.deserialize(data.brainState);
            if (state.brain) state.brain.expandVocab(state.vocab.length);
            if (data.currentLoss !== undefined) state.currentLoss = data.currentLoss;
            if (data.iterations !== undefined) state.iterations = data.iterations;
        }
        renderCorpus();
        if (DOM.uiLoss) DOM.uiLoss.innerText = state.currentLoss.toFixed(5);
        if (DOM.cloudStatus) {
            DOM.cloudStatus.innerHTML = `<i data-lucide="hard-drive" class="w-3 h-3"></i> Local Storage`;
            DOM.cloudStatus.classList.replace('text-amber-500', 'text-green-500');
            if (window.lucide) window.lucide.createIcons();
        }
    } catch (e) {
        console.error("Error loading data from localStorage", e);
        // Ensure UI is initialized even if localStorage fails
        renderCorpus();
        if (DOM.uiLoss) DOM.uiLoss.innerText = state.currentLoss.toFixed(5);
        if (DOM.cloudStatus) {
            DOM.cloudStatus.innerHTML = `<i data-lucide="cloud-off" class="w-3 h-3"></i> New Local Mode`;
            if (window.lucide) window.lucide.createIcons();
        }
    }
}

export function saveData() {
    try {
        const dataObj = {
            vocab: state.vocab,
            corpus: state.corpus,
            brainState: state.brain ? state.brain.serialize() : null,
            currentLoss: state.currentLoss,
            iterations: state.iterations
        };
        localStorage.setItem('neuralEngineData', JSON.stringify(dataObj));
    } catch (e) {
        console.error("Error saving data to localStorage", e);
    }
}
