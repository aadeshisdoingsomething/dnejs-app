export class AdvancedBrain {
    constructor(vocabSize, hiddenNodes) {
        this.vocabSize = vocabSize;
        this.hiddenNodes = hiddenNodes;
        this.learningRate = 0.2;
        this.momentum = 0.9;
        this.initWeights();
    }

    initWeights() {
        const limitIH = Math.sqrt(2 / this.vocabSize); // He variance for ReLU
        const limitHO = Math.sqrt(1 / this.hiddenNodes); // Xavier for Sigmoid

        const lenIH = this.hiddenNodes * this.vocabSize;
        const lenHO = this.vocabSize * this.hiddenNodes;

        this.weightsIH = new Float32Array(lenIH);
        this.weightsHO = new Float32Array(lenHO);
        this.biasH = new Float32Array(this.hiddenNodes);
        this.biasO = new Float32Array(this.vocabSize);

        this.vWeightsIH = new Float32Array(lenIH);
        this.vWeightsHO = new Float32Array(lenHO);
        this.vBiasH = new Float32Array(this.hiddenNodes);
        this.vBiasO = new Float32Array(this.vocabSize);

        this.gradWeightsIH = new Float32Array(lenIH);
        this.gradWeightsHO = new Float32Array(lenHO);
        this.gradBiasH = new Float32Array(this.hiddenNodes);
        this.gradBiasO = new Float32Array(this.vocabSize);

        for (let i = 0; i < lenIH; i++) {
            this.weightsIH[i] = (Math.random() * 2 - 1) * limitIH;
        }
        for (let i = 0; i < lenHO; i++) {
            this.weightsHO[i] = (Math.random() * 2 - 1) * limitHO;
        }
    }

    expandVocab(newSize) {
        const diff = newSize - this.vocabSize;
        if (diff <= 0) return;

        const limitIH = Math.sqrt(2 / newSize);
        const limitHO = Math.sqrt(1 / this.hiddenNodes);

        const newLenIH = this.hiddenNodes * newSize;
        const newLenHO = newSize * this.hiddenNodes;

        const newWeightsIH = new Float32Array(newLenIH);
        const newWeightsHO = new Float32Array(newLenHO);
        const newBiasO = new Float32Array(newSize);

        const newVWeightsIH = new Float32Array(newLenIH);
        const newVWeightsHO = new Float32Array(newLenHO);
        const newVBiasO = new Float32Array(newSize);

        const newGradWeightsIH = new Float32Array(newLenIH);
        const newGradWeightsHO = new Float32Array(newLenHO);
        const newGradBiasO = new Float32Array(newSize);
        // Copy old data and initialize new values
        for (let i = 0; i < this.hiddenNodes; i++) {
            for (let j = 0; j < this.vocabSize; j++) {
                newWeightsIH[i * newSize + j] = this.weightsIH[i * this.vocabSize + j];
                newVWeightsIH[i * newSize + j] = this.vWeightsIH[i * this.vocabSize + j];
                newGradWeightsIH[i * newSize + j] = this.gradWeightsIH[i * this.vocabSize + j];
            }
            for (let j = this.vocabSize; j < newSize; j++) {
                newWeightsIH[i * newSize + j] = (Math.random() * 2 - 1) * limitIH;
            }
        }

        for (let i = 0; i < this.vocabSize; i++) {
            for (let j = 0; j < this.hiddenNodes; j++) {
                newWeightsHO[i * this.hiddenNodes + j] = this.weightsHO[i * this.hiddenNodes + j];
                newVWeightsHO[i * this.hiddenNodes + j] = this.vWeightsHO[i * this.hiddenNodes + j];
                newGradWeightsHO[i * this.hiddenNodes + j] = this.gradWeightsHO[i * this.hiddenNodes + j];
            }
        }
        for (let i = this.vocabSize; i < newSize; i++) {
            for (let j = 0; j < this.hiddenNodes; j++) {
                newWeightsHO[i * this.hiddenNodes + j] = (Math.random() * 2 - 1) * limitHO;
            }
        }

        newBiasO.set(this.biasO);
        newVBiasO.set(this.vBiasO);
        newGradBiasO.set(this.gradBiasO);

        this.weightsIH = newWeightsIH;
        this.weightsHO = newWeightsHO;
        this.biasO = newBiasO;

        this.vWeightsIH = newVWeightsIH;
        this.vWeightsHO = newVWeightsHO;
        this.vBiasO = newVBiasO;

        this.gradWeightsIH = newGradWeightsIH;
        this.gradWeightsHO = newGradWeightsHO;
        this.gradBiasO = newGradBiasO;

        this.vocabSize = newSize;
    }

    sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
    dsigmoid(y) { return y * (1 - y); }
    relu(x) { return x > 0 ? x : 0.01 * x; }
    drelu(y) { return y > 0 ? 1 : 0.01; }

    predict(inputVector) {
        let hidden = new Array(this.hiddenNodes);
        for (let i = 0; i < this.hiddenNodes; i++) {
            let sum = this.biasH[i];
            for (let j = 0; j < this.vocabSize; j++) {
                sum += this.weightsIH[i * this.vocabSize + j] * (inputVector[j] || 0);
            }
            hidden[i] = this.relu(sum);
        }

        let output = new Array(this.vocabSize);
        let maxSum = -Infinity;
        let sums = new Array(this.vocabSize);

        // Calculate sums and find max for numerical stability
        for (let i = 0; i < this.vocabSize; i++) {
            let sum = this.biasO[i];
            for (let j = 0; j < this.hiddenNodes; j++) {
                sum += this.weightsHO[i * this.hiddenNodes + j] * hidden[j];
            }
            sums[i] = sum;
            if (sum > maxSum) maxSum = sum;
        }

        // Apply Softmax: exp(val) / sum(exp(val))
        let totalExp = 0;
        for (let i = 0; i < this.vocabSize; i++) {
            // Clamp exponent to avoid NaN Infinity
            let exponent = Math.min(Math.max(sums[i] - maxSum, -50), 50);
            let e = Math.exp(exponent);
            output[i] = e;
            totalExp += e;
        }
        for (let i = 0; i < this.vocabSize; i++) {
            output[i] /= totalExp;
        }

        return { output, hidden };
    }

    resetGradients() {
        this.gradWeightsIH.fill(0);
        this.gradWeightsHO.fill(0);
        this.gradBiasH.fill(0);
        this.gradBiasO.fill(0);
    }

    accumulateGradients(inputVector, targetVector) {
        const { output, hidden } = this.predict(inputVector);
        const outputErrors = targetVector.map((target, i) => target - output[i]);

        const hiddenErrors = new Array(this.hiddenNodes).fill(0);
        for (let i = 0; i < this.vocabSize; i++) {
            for (let j = 0; j < this.hiddenNodes; j++) {
                hiddenErrors[j] += outputErrors[i] * this.weightsHO[i * this.hiddenNodes + j];
            }
        }

        // Gradient Clipping Threshold
        const clip = 5.0;

        for (let i = 0; i < this.vocabSize; i++) {
            let gradient = outputErrors[i]; // Softmax + CrossEntropy derivative
            // Clip Error Gradient
            gradient = Math.max(-clip, Math.min(clip, gradient));

            for (let j = 0; j < this.hiddenNodes; j++) {
                this.gradWeightsHO[i * this.hiddenNodes + j] += gradient * hidden[j];
            }
            this.gradBiasO[i] += gradient;
        }

        for (let i = 0; i < this.hiddenNodes; i++) {
            let gradient = hiddenErrors[i] * this.drelu(hidden[i]);
            // Clip Hidden Gradient
            gradient = Math.max(-clip, Math.min(clip, gradient));

            for (let j = 0; j < this.vocabSize; j++) {
                this.gradWeightsIH[i * this.vocabSize + j] += gradient * (inputVector[j] || 0);
            }
            this.gradBiasH[i] += gradient;
        }

        let ceLoss = 0;
        for (let i = 0; i < this.vocabSize; i++) {
            const t = targetVector[i];
            const o = Math.max(1e-15, Math.min(1 - 1e-15, output[i]));
            ceLoss -= (t * Math.log(o) + (1 - t) * Math.log(1 - o));
        }
        return ceLoss;
    }

    applyGradients(batchSize, customLR) {
        const lr = (customLR || this.learningRate) / batchSize;

        for (let i = 0; i < this.vocabSize; i++) {
            for (let j = 0; j < this.hiddenNodes; j++) {
                let hoIdx = i * this.hiddenNodes + j;
                let delta = this.gradWeightsHO[hoIdx] * lr;
                this.vWeightsHO[hoIdx] = this.momentum * this.vWeightsHO[hoIdx] + delta;
                this.weightsHO[hoIdx] += this.vWeightsHO[hoIdx];
            }
            this.vBiasO[i] = this.momentum * this.vBiasO[i] + this.gradBiasO[i] * lr;
            this.biasO[i] += this.vBiasO[i];
        }

        for (let i = 0; i < this.hiddenNodes; i++) {
            for (let j = 0; j < this.vocabSize; j++) {
                let ihIdx = i * this.vocabSize + j;
                let delta = this.gradWeightsIH[ihIdx] * lr;
                this.vWeightsIH[ihIdx] = this.momentum * this.vWeightsIH[ihIdx] + delta;
                this.weightsIH[ihIdx] += this.vWeightsIH[ihIdx];
            }
            this.vBiasH[i] = this.momentum * this.vBiasH[i] + this.gradBiasH[i] * lr;
            this.biasH[i] += this.vBiasH[i];
        }

        this.resetGradients();
    }

    serialize() {
        return JSON.stringify({
            vocabSize: this.vocabSize,
            hiddenNodes: this.hiddenNodes,
            weightsIH: Array.from(this.weightsIH),
            weightsHO: Array.from(this.weightsHO),
            biasH: Array.from(this.biasH),
            biasO: Array.from(this.biasO),
            vWeightsIH: Array.from(this.vWeightsIH),
            vWeightsHO: Array.from(this.vWeightsHO),
            vBiasH: Array.from(this.vBiasH),
            vBiasO: Array.from(this.vBiasO)
        });
    }

    deserialize(dataStr) {
        const data = JSON.parse(dataStr);
        this.vocabSize = data.vocabSize;
        this.hiddenNodes = data.hiddenNodes;
        this.weightsIH = new Float32Array(data.weightsIH);
        this.weightsHO = new Float32Array(data.weightsHO);
        this.biasH = new Float32Array(data.biasH);
        this.biasO = new Float32Array(data.biasO);
        this.vWeightsIH = new Float32Array(data.vWeightsIH || new Array(this.hiddenNodes * this.vocabSize).fill(0));
        this.vWeightsHO = new Float32Array(data.vWeightsHO || new Array(this.vocabSize * this.hiddenNodes).fill(0));
        this.vBiasH = new Float32Array(data.vBiasH || new Array(this.hiddenNodes).fill(0));
        this.vBiasO = new Float32Array(data.vBiasO || new Array(this.vocabSize).fill(0));

        // Re-initialize gradient buffers if deserializing an old save without them
        const lenIH = this.hiddenNodes * this.vocabSize;
        const lenHO = this.vocabSize * this.hiddenNodes;
        this.gradWeightsIH = new Float32Array(lenIH);
        this.gradWeightsHO = new Float32Array(lenHO);
        this.gradBiasH = new Float32Array(this.hiddenNodes);
        this.gradBiasO = new Float32Array(this.vocabSize);
    }

    getRawState() {
        return {
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
        };
    }

    setRawState(data) {
        this.vocabSize = data.vocabSize;
        this.hiddenNodes = data.hiddenNodes;
        this.weightsIH = data.weightsIH;
        this.weightsHO = data.weightsHO;
        this.biasH = data.biasH;
        this.biasO = data.biasO;
        this.vWeightsIH = data.vWeightsIH;
        this.vWeightsHO = data.vWeightsHO;
        this.vBiasH = data.vBiasH;
        this.vBiasO = data.vBiasO;

        // Ensure gradient arrays match the incoming size if missing
        const lenIH = this.hiddenNodes * this.vocabSize;
        if (!this.gradWeightsIH || this.gradWeightsIH.length !== lenIH) {
            const lenHO = this.vocabSize * this.hiddenNodes;
            this.gradWeightsIH = new Float32Array(lenIH);
            this.gradWeightsHO = new Float32Array(lenHO);
            this.gradBiasH = new Float32Array(this.hiddenNodes);
            this.gradBiasO = new Float32Array(this.vocabSize);
        }
    }
}
