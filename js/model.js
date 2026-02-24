export class AdvancedBrain {
    constructor(vocabSize, hiddenNodes) {
        this.vocabSize = vocabSize;
        this.hiddenNodes = hiddenNodes;
        this.learningRate = 0.2;
        this.momentum = 0.9;
        this.initWeights();
    }

    initWeights() {
        const limitIH = Math.sqrt(6 / (this.vocabSize + this.hiddenNodes));
        const limitHO = Math.sqrt(6 / (this.hiddenNodes + this.vocabSize));

        this.weightsIH = Array.from({ length: this.hiddenNodes }, () => Array.from({ length: this.vocabSize }, () => (Math.random() * 2 - 1) * limitIH));
        this.weightsHO = Array.from({ length: this.vocabSize }, () => Array.from({ length: this.hiddenNodes }, () => (Math.random() * 2 - 1) * limitHO));
        this.biasH = new Array(this.hiddenNodes).fill(0);
        this.biasO = new Array(this.vocabSize).fill(0);

        this.vWeightsIH = Array.from({ length: this.hiddenNodes }, () => new Array(this.vocabSize).fill(0));
        this.vWeightsHO = Array.from({ length: this.vocabSize }, () => new Array(this.hiddenNodes).fill(0));
        this.vBiasH = new Array(this.hiddenNodes).fill(0);
        this.vBiasO = new Array(this.vocabSize).fill(0);
    }

    expandVocab(newSize) {
        const diff = newSize - this.vocabSize;
        if (diff <= 0) return;

        const limitIH = Math.sqrt(6 / (newSize + this.hiddenNodes));
        const limitHO = Math.sqrt(6 / (this.hiddenNodes + newSize));

        this.weightsIH = this.weightsIH.map(row => [...row, ...Array.from({ length: diff }, () => (Math.random() * 2 - 1) * limitIH)]);
        const newRowsHO = Array.from({ length: diff }, () => Array.from({ length: this.hiddenNodes }, () => (Math.random() * 2 - 1) * limitHO));
        this.weightsHO = [...this.weightsHO, ...newRowsHO];

        this.biasO = [...this.biasO, ...new Array(diff).fill(0)];

        this.vWeightsIH = this.vWeightsIH.map(row => [...row, ...new Array(diff).fill(0)]);
        const newVRowsHO = Array.from({ length: diff }, () => new Array(this.hiddenNodes).fill(0));
        this.vWeightsHO = [...this.vWeightsHO, ...newVRowsHO];
        this.vBiasO = [...this.vBiasO, ...new Array(diff).fill(0)];

        this.vocabSize = newSize;
    }

    sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    dsigmoid(y) { return y * (1 - y); }

    predict(inputVector) {
        let hidden = this.biasH.map((b, i) => {
            let sum = b;
            for (let j = 0; j < this.vocabSize; j++) {
                sum += (this.weightsIH[i][j] || 0) * (inputVector[j] || 0);
            }
            return this.sigmoid(sum);
        });

        let output = this.biasO.map((b, i) => {
            let sum = b;
            for (let j = 0; j < this.hiddenNodes; j++) {
                sum += (this.weightsHO[i][j] || 0) * hidden[j];
            }
            return this.sigmoid(sum);
        });

        return { output, hidden };
    }

    train(inputVector, targetVector, customLR) {
        const lr = customLR || this.learningRate;
        const { output, hidden } = this.predict(inputVector);
        const outputErrors = targetVector.map((target, i) => target - output[i]);

        const hiddenErrors = new Array(this.hiddenNodes).fill(0);
        for (let i = 0; i < this.vocabSize; i++) {
            for (let j = 0; j < this.hiddenNodes; j++) {
                hiddenErrors[j] += outputErrors[i] * this.weightsHO[i][j];
            }
        }

        for (let i = 0; i < this.vocabSize; i++) {
            let gradient = outputErrors[i] * this.dsigmoid(output[i]) * lr;
            for (let j = 0; j < this.hiddenNodes; j++) {
                let delta = gradient * hidden[j];
                this.vWeightsHO[i][j] = this.momentum * this.vWeightsHO[i][j] + delta;
                this.weightsHO[i][j] += this.vWeightsHO[i][j];
            }
            this.vBiasO[i] = this.momentum * this.vBiasO[i] + gradient;
            this.biasO[i] += this.vBiasO[i];
        }

        for (let i = 0; i < this.hiddenNodes; i++) {
            let gradient = hiddenErrors[i] * this.dsigmoid(hidden[i]) * lr;
            for (let j = 0; j < this.vocabSize; j++) {
                let delta = gradient * (inputVector[j] || 0);
                this.vWeightsIH[i][j] = this.momentum * this.vWeightsIH[i][j] + delta;
                this.weightsIH[i][j] += this.vWeightsIH[i][j];
            }
            this.vBiasH[i] = this.momentum * this.vBiasH[i] + gradient;
            this.biasH[i] += this.vBiasH[i];
        }

        return outputErrors.reduce((sum, err) => sum + Math.abs(err), 0);
    }

    serialize() {
        return JSON.stringify({
            vocabSize: this.vocabSize,
            hiddenNodes: this.hiddenNodes,
            weightsIH: this.weightsIH,
            weightsHO: this.weightsHO,
            biasH: this.biasH,
            biasO: this.biasO,
            vWeightsIH: this.vWeightsIH,
            vWeightsHO: this.vWeightsHO,
            vBiasH: this.vBiasH,
            vBiasO: this.vBiasO
        });
    }

    deserialize(dataStr) {
        const data = JSON.parse(dataStr);
        this.vocabSize = data.vocabSize;
        this.hiddenNodes = data.hiddenNodes;
        this.weightsIH = data.weightsIH;
        this.weightsHO = data.weightsHO;
        this.biasH = data.biasH;
        this.biasO = data.biasO;
        this.vWeightsIH = data.vWeightsIH || this.vWeightsIH;
        this.vWeightsHO = data.vWeightsHO || this.vWeightsHO;
        this.vBiasH = data.vBiasH || this.vBiasH;
        this.vBiasO = data.vBiasO || this.vBiasO;
    }
}
