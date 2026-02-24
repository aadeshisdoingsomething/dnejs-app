import { state } from './state.js';

export const DOM = {
  get uiLoss() { return document.getElementById('ui-loss'); },
  get uiTargetLoss() { return document.getElementById('ui-target-loss'); },
  get btnTrain() { return document.getElementById('btn-train'); },
  get trainIcon() { return document.getElementById('train-icon'); },
  get trainText() { return document.getElementById('train-text'); },
  get chatMessages() { return document.getElementById('chat-messages'); },
  get chatInput() { return document.getElementById('chat-input'); },
  get cloudStatus() { return document.getElementById('cloud-status'); },
  get corpusList() { return document.getElementById('corpus-list'); },
  get corpusStats() { return document.getElementById('corpus-stats'); },
  get chatForm() { return document.getElementById('chat-form'); },
  get tabManual() { return document.getElementById('tab-manual'); },
  get tabJson() { return document.getElementById('tab-json'); },
  get viewManual() { return document.getElementById('view-manual'); },
  get viewJson() { return document.getElementById('view-json'); },
  get btnTeach() { return document.getElementById('btn-teach'); },
  get teachQ() { return document.getElementById('teach-q'); },
  get teachA() { return document.getElementById('teach-a'); },
  get jsonInput() { return document.getElementById('json-input'); },
  get jsonError() { return document.getElementById('json-error'); },
  get btnImportJson() { return document.getElementById('btn-import-json'); },
  get btnPasteExample() { return document.getElementById('btn-paste-example'); },
  get visOverlay() { return document.getElementById('visualizer-overlay'); },
  get visContent() { return document.getElementById('vis-content'); },
  get btnVisualizer() { return document.getElementById('btn-visualizer'); },
  get btnCloseVis() { return document.getElementById('btn-close-vis'); },
  get btnClearChat() { return document.getElementById('btn-clear-chat'); },
  get btnNukeData() { return document.getElementById('btn-nuke-data'); }
};

export function renderCorpus() {
  DOM.corpusStats.innerText = `Corpus (${state.corpus.length}) / Vocab (${state.vocab.length})`;
  DOM.corpusList.innerHTML = state.corpus.map(c => `
        <div class="text-[10px] text-neutral-400 bg-black/20 p-2 rounded-lg border border-neutral-800/50 truncate">
          <span class="text-indigo-500 font-bold">Q:</span> ${c.q} <span class="text-neutral-600 mx-1">|</span> <span class="text-green-500/70 font-bold">A:</span> ${c.a}
        </div>
      `).join('');
}

export function appendMessage(role, text, msgIndex) {
  const isUser = role === 'user';
  const div = document.createElement('div');
  div.className = `flex flex-col ${isUser ? 'items-end' : 'items-start'}`;

  let html = `
        <div class="max-w-[85%] px-6 py-4 rounded-3xl text-sm leading-relaxed ${isUser ? 'bg-indigo-600 text-white rounded-tr-none shadow-lg shadow-indigo-500/10' : 'bg-neutral-800 text-neutral-100 border border-neutral-700 rounded-tl-none'}">
          ${text}
        </div>
      `;

  if (!isUser) {
    html += `
          <div class="flex gap-2 mt-2 ml-2 opacity-0 hover:opacity-100 transition-opacity">
            <button onclick="window.handleFeedback(${msgIndex}, true)" class="p-1 text-neutral-600 hover:text-green-500"><i data-lucide="thumbs-up" class="w-[14px] h-[14px]"></i></button>
            <button onclick="window.handleFeedback(${msgIndex}, false)" class="p-1 text-neutral-600 hover:text-rose-500"><i data-lucide="thumbs-down" class="w-[14px] h-[14px]"></i></button>
          </div>
        `;
  }

  div.innerHTML = html;
  DOM.chatMessages.appendChild(div);
  if (window.lucide) window.lucide.createIcons();
  DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
}

export function updateVisualizer() {
  let html = `
        <div class="space-y-2">
          <h3 class="text-xs font-bold text-neutral-400 uppercase tracking-widest border-b border-neutral-800 pb-2">Input → Hidden Weights (IH)</h3>
          <p class="text-[10px] text-neutral-600 mb-4">Rows: Hidden (${state.brain.hiddenNodes}) | Cols: Vocab (${state.brain.vocabSize})</p>
          <div class="flex flex-col gap-1">
            ${state.brain.weightsIH.map((row, i) => `
              <div class="flex gap-1">
                ${row.map((val, j) => `
                  <div class="w-4 h-4 rounded-[2px]" style="background-color: ${val > 0 ? `rgba(99, 102, 241, ${Math.min(Math.abs(val) * 2, 1)})` : `rgba(244, 63, 94, ${Math.min(Math.abs(val) * 2, 1)})`}" title="Node ${i} -> Vocab[${state.vocab[j]}]: ${val.toFixed(3)}"></div>
                `).join('')}
              </div>
            `).join('')}
          </div>
        </div>

        <div class="space-y-2">
          <h3 class="text-xs font-bold text-neutral-400 uppercase tracking-widest border-b border-neutral-800 pb-2">Hidden → Output Weights (HO)</h3>
          <p class="text-[10px] text-neutral-600 mb-4">Rows: Vocab (${state.brain.vocabSize}) | Cols: Hidden (${state.brain.hiddenNodes})</p>
          <div class="flex flex-col gap-1">
            ${state.brain.weightsHO.map((row, i) => `
              <div class="flex gap-1 items-center">
                <span class="text-[9px] w-12 truncate text-right text-neutral-500 mr-2">${state.vocab[i]}</span>
                ${row.map((val, j) => `
                  <div class="w-4 h-4 rounded-[2px]" style="background-color: ${val > 0 ? `rgba(34, 197, 94, ${Math.min(Math.abs(val) * 2, 1)})` : `rgba(234, 179, 8, ${Math.min(Math.abs(val) * 2, 1)})`}" title="Vocab[${state.vocab[i]}] <- Node ${j}: ${val.toFixed(3)}"></div>
                `).join('')}
              </div>
            `).join('')}
          </div>
        </div>
      `;
  DOM.visContent.innerHTML = html;
}
