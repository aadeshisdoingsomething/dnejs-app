import { state } from './state.js';
import { DOM, renderCorpus } from './ui.js';

const DB_NAME = 'NeuralEngineDB';
const STORE_NAME = 'engineState';
const DB_VERSION = 1;

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        request.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
    });
}

function updateStorageUI(byteSize) {
    if (DOM.cloudStatus) {
        DOM.cloudStatus.innerHTML = `<i data-lucide="hard-drive" class="w-3 h-3"></i> IndexedDB (${formatBytes(byteSize)})`;
        DOM.cloudStatus.classList.replace('text-amber-500', 'text-green-500');
        if (window.lucide) window.lucide.createIcons();
    }
}

export async function loadData() {
    try {
        // Attempt Migration from Legacy localStorage
        const legacyDataStr = localStorage.getItem('neuralEngineData');
        if (legacyDataStr) {
            console.log("Migrating legacy localStorage to IndexedDB...");
            const legacyData = JSON.parse(legacyDataStr);
            await saveToIndexedDB(legacyData);
            localStorage.removeItem('neuralEngineData');
        }

        const db = await openDB();
        const transaction = db.transaction(STORE_NAME, 'readonly');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.get('coreData');

        const dataStr = await new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });

        if (dataStr) {
            const byteSize = new Blob([dataStr]).size;
            updateStorageUI(byteSize);

            const data = JSON.parse(dataStr);
            if (data.vocab) state.vocab = data.vocab;
            if (data.corpus) state.corpus = data.corpus;

            try {
                if (data.brainState && state.brain) state.brain.deserialize(data.brainState);
            } catch (e) {
                console.error("Critical Model Geometry Mismatch:", e);
                console.warn("Auto-Purging Storage Database to restore Engine Function...");
                indexedDB.deleteDatabase(DB_NAME);
                localStorage.removeItem('neuralEngineData');
                alert("Architecture Upgrade Detected. Clearing incompatible legacy brain structures. The page will now reload.");
                window.location.reload();
                return;
            }

            if (state.brain) state.brain.expandVocab(state.vocab.length);
            if (data.currentLoss !== undefined) state.currentLoss = data.currentLoss;
            if (data.iterations !== undefined) state.iterations = data.iterations;
        } else {
            updateStorageUI(0);
        }

        renderCorpus();
        if (DOM.uiLoss) DOM.uiLoss.innerText = state.currentLoss.toFixed(5);

    } catch (e) {
        console.error("Error loading data from IndexedDB", e);
        renderCorpus();
        if (DOM.uiLoss) DOM.uiLoss.innerText = state.currentLoss.toFixed(5);
        if (DOM.cloudStatus) {
            DOM.cloudStatus.innerHTML = `<i data-lucide="cloud-off" class="w-3 h-3"></i> New Local Mode`;
            if (window.lucide) window.lucide.createIcons();
        }
    }
}

function saveToIndexedDB(dataObj) {
    return new Promise(async (resolve, reject) => {
        try {
            const db = await openDB();
            const transaction = db.transaction(STORE_NAME, 'readwrite');
            const store = transaction.objectStore(STORE_NAME);

            const dataStr = JSON.stringify(dataObj);
            const byteSize = new Blob([dataStr]).size;

            const request = store.put(dataStr, 'coreData');

            request.onsuccess = () => {
                updateStorageUI(byteSize);
                resolve();
            };
            request.onerror = () => reject(request.error);
        } catch (e) {
            reject(e);
        }
    });
}

export async function saveData() {
    try {
        const dataObj = {
            vocab: state.vocab,
            corpus: state.corpus,
            brainState: state.brain ? state.brain.serialize() : null,
            currentLoss: state.currentLoss,
            iterations: state.iterations
        };
        await saveToIndexedDB(dataObj);
    } catch (e) {
        console.error("Error saving data to IndexedDB", e);
    }
}
