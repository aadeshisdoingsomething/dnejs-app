import { state } from './state.js';
import { loadData, saveData } from './storage.js';
import { DOM, renderCorpus, appendMessage, updateVisualizer } from './ui.js';
import { toggleTraining } from './training.js';
import { encodeSequence, decode, tokenize, updateEncodedCorpus } from './nlp.js';

window.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) window.lucide.createIcons();
    loadData();
    updateEncodedCorpus();

    window.handleFeedback = function (index, positive) {
        // Feedback mechanism temporarily disabled during sequence migration
        // Will need to be rewritten to support sequential reinforcement in the future
        console.warn("Feedback training currently disabled for Auto-Regressive model.");
    };

    DOM.chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = DOM.chatInput.value.trim();
        if (!text) return;

        state.messages.push({ role: 'user', text });
        appendMessage('user', text, state.messages.length - 1);

        // Auto-Regressive Inference Loop
        let currentContext = tokenize(text);
        let generatedWords = [];
        let maxTokens = 20; // Hard limit to prevent infinite loops on untrained models

        for (let i = 0; i < maxTokens; i++) {
            const vec = encodeSequence(currentContext);
            const pred = state.brain.predict(vec);
            const word = decode(pred.output);

            if (word === "<EOS>") break;

            generatedWords.push(word);
            currentContext.push(word); // Feed prediction back into sequence
        }

        const responseMsg = generatedWords.length > 0 ? generatedWords.join(" ") : "...";

        state.messages.push({ role: 'ai', text: responseMsg });
        appendMessage('ai', responseMsg, state.messages.length - 1);

        DOM.chatInput.value = '';
    });

    DOM.btnTrain.addEventListener('click', () => toggleTraining());

    DOM.tabManual.addEventListener('click', () => {
        DOM.tabManual.className = "flex-1 py-2 rounded-lg text-xs font-bold transition-colors bg-indigo-600/20 text-indigo-400";
        DOM.tabJson.className = "flex-1 py-2 rounded-lg text-xs font-bold transition-colors bg-neutral-800 text-neutral-500 hover:text-neutral-300";
        DOM.viewManual.classList.remove('hidden');
        DOM.viewJson.classList.add('hidden');
    });

    DOM.tabJson.addEventListener('click', () => {
        DOM.tabJson.className = "flex-1 py-2 rounded-lg text-xs font-bold transition-colors bg-indigo-600/20 text-indigo-400";
        DOM.tabManual.className = "flex-1 py-2 rounded-lg text-xs font-bold transition-colors bg-neutral-800 text-neutral-500 hover:text-neutral-300";
        DOM.viewJson.classList.remove('hidden');
        DOM.viewManual.classList.add('hidden');
    });

    DOM.btnTeach.addEventListener('click', () => {
        const q = DOM.teachQ.value;
        const a = DOM.teachA.value;
        if (!q || !a) return;

        const newWords = [...tokenize(q), ...tokenize(a)];
        const uniqueNewWords = newWords.filter(w => !state.vocab.includes(w));
        if (uniqueNewWords.length > 0) {
            state.vocab.push(...uniqueNewWords);
            state.brain.expandVocab(state.vocab.length);
        }
        state.corpus.push({ q, a });
        updateEncodedCorpus();
        renderCorpus();
        saveData();

        DOM.teachQ.value = '';
        DOM.teachA.value = '';
    });

    DOM.btnPasteExample.addEventListener('click', () => {
        DOM.jsonInput.value = `[\n  { "q": "what is your name", "a": "i am dynamic brain" },\n  { "q": "tell me a joke", "a": "math is no joke" }\n]`;
    });

    DOM.btnImportJson.addEventListener('click', () => {
        const input = DOM.jsonInput.value;
        try {
            DOM.jsonError.classList.add('hidden');
            const parsed = JSON.parse(input);
            if (!Array.isArray(parsed)) throw new Error("Root must be array");

            let newWords = [];
            let validPairs = [];
            parsed.forEach(item => {
                if (!item.q || !item.a) throw new Error("Missing q or a strings");
                newWords.push(...tokenize(item.q), ...tokenize(item.a));
                validPairs.push({ q: item.q, a: item.a });
            });

            const uniqueNewWords = newWords.filter(w => !state.vocab.includes(w));
            if (uniqueNewWords.length > 0) {
                state.vocab.push(...uniqueNewWords);
                state.brain.expandVocab(state.vocab.length);
            }
            state.corpus.push(...validPairs);
            updateEncodedCorpus();
            renderCorpus();
            saveData();

            DOM.jsonInput.value = '';
            DOM.tabManual.click();
        } catch (err) {
            DOM.jsonError.innerText = err.message;
            DOM.jsonError.classList.remove('hidden');
        }
    });

    DOM.btnVisualizer.addEventListener('click', () => {
        DOM.visOverlay.classList.remove('hidden');
        updateVisualizer();
    });

    DOM.btnCloseVis.addEventListener('click', () => {
        DOM.visOverlay.classList.add('hidden');
    });

    DOM.btnClearChat.addEventListener('click', () => {
        DOM.chatMessages.innerHTML = '';
        state.messages = [];
    });

    DOM.btnNukeData.addEventListener('click', () => {
        if (confirm("Are you sure you want to NUKE the core data? This will clear all training, words, and matrices, and cannot be undone.")) {
            localStorage.removeItem('neuralEngineData');
            window.location.reload();
        }
    });
});
