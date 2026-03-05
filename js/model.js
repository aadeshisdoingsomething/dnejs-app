import * as tf from 'https://esm.sh/@tensorflow/tfjs@4.22.0';

export class AdvancedBrain {
    constructor(vocabSize, hiddenNodes, contextWindowSize = 15) {
        this.vocabSize = vocabSize;
        this.hiddenNodes = hiddenNodes;
        this.contextWindowSize = contextWindowSize; // Critical for fixed-length Dense Embeddings
        // Switch to Adam targets to prevent Overshooting divergence
        this.learningRate = 0.01;
        this.initModel();
    }

    initModel() {
        this.model = tf.sequential();

        // Embedding Layer: vocabSize -> 16-Dimensional Semantic Space
        // Takes flat integer arrays (e.g. [14, 25, 3]) and mathematically clusters words
        this.model.add(tf.layers.embedding({
            inputDim: this.vocabSize,
            outputDim: 16,
            inputLength: this.contextWindowSize || 15
        }));

        // Flatten the 2D Semantic Matrix back to 1D for Dense Layers
        // If inputLength isn't explicitly defined, Flatten crashes.
        this.model.add(tf.layers.flatten());

        // Flatten -> Hidden 1: hiddenNodes (Tanh)
        this.model.add(tf.layers.dense({
            units: this.hiddenNodes,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));

        // Hidden 1: hiddenNodes -> Hidden 2: hiddenNodes (Tanh)
        this.model.add(tf.layers.dense({
            units: this.hiddenNodes,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));

        // Hidden 2: hiddenNodes -> Output: vocabSize (Softmax)
        this.model.add(tf.layers.dense({
            units: this.vocabSize,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));

        this.model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'categoricalCrossentropy'
        });
    }

    expandVocab(newSize) {
        if (newSize <= this.vocabSize) return;

        const oldWeights = this.model.getWeights();
        // Index 0: Embedding Matrix [oldVocabSize, 16]
        // Index 1: Flatten->H1 Kernel [(16 * contextWindowSize), hiddenNodes]
        // Index 2: Flatten->H1 Bias   [hiddenNodes]
        // Index 3: H1->H2 Kernel [hiddenNodes, hiddenNodes]
        // Index 4: H1->H2 Bias   [hiddenNodes]
        // Index 5: H2->Output Kernel [hiddenNodes, oldVocabSize]
        // Index 6: H2->Output Bias   [oldVocabSize]

        const newModel = tf.sequential();
        newModel.add(tf.layers.embedding({
            inputDim: newSize,
            outputDim: 16,
            inputLength: this.contextWindowSize || 15
        }));
        newModel.add(tf.layers.flatten());
        newModel.add(tf.layers.dense({
            units: this.hiddenNodes,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));
        newModel.add(tf.layers.dense({
            units: this.hiddenNodes,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));
        newModel.add(tf.layers.dense({
            units: newSize,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));

        newModel.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'categoricalCrossentropy'
        });

        tf.tidy(() => {
            // Expansion Pads
            const paddingRows = newSize - this.vocabSize;

            // 1. Embedding Kernel Pad
            // Pad the vocabulary but keep the output dimension strictly at 16
            const embeddingPad = tf.randomNormal([paddingRows, 16], 0, Math.sqrt(2 / newSize));
            const newEmbeddingMatrix = tf.concat([oldWeights[0], embeddingPad], 0);

            // 2. Flatten -> H1 Kernel (Unchanged - geometry is bound to 16 * contextWindowSize)
            const newFlattenH1Kernel = oldWeights[1].clone();
            const newFlattenH1Bias = oldWeights[2].clone();

            // 3. H1 -> H2 Kernel & Bias (Untouched)
            const newH1H2Kernel = oldWeights[3].clone();
            const newH1H2Bias = oldWeights[4].clone();

            // 4. H2 -> Output Kernel Pad
            const hoKernelPad = tf.randomNormal([this.hiddenNodes, paddingRows], 0, Math.sqrt(1 / this.hiddenNodes));
            const newHoKernel = tf.concat([oldWeights[5], hoKernelPad], 1);

            // 5. H2 -> Output Bias Pad
            const hoBiasPad = tf.zeros([paddingRows]);
            const newHoBias = tf.concat([oldWeights[6], hoBiasPad], 0);

            newModel.setWeights([newEmbeddingMatrix, newFlattenH1Kernel, newFlattenH1Bias, newH1H2Kernel, newH1H2Bias, newHoKernel, newHoBias]);
        });

        this.model.dispose(); // Releases old tensors
        this.model = newModel;
        this.vocabSize = newSize;
    }

    predict(inputVector) {
        return tf.tidy(() => {
            // Embeddings take flat integer sequences instead of One-Hot arrays
            // inputVector = [14, 25, 3] instead of [0, 0, 1.4, 0, 0]
            const inputTensor = tf.tensor2d([inputVector], [1, this.contextWindowSize]);
            const pred = this.model.predict(inputTensor);
            const outputArray = Array.from(pred.dataSync());
            return { output: outputArray, hidden: [] };
        });
    }

    async trainBatch(encodedPairs, batchSize = 100) {
        let lossValue = 0;

        const xArray = [];
        const yArray = [];

        for (let k = 0; k < batchSize; k++) {
            const sample = encodedPairs[Math.floor(Math.random() * encodedPairs.length)];
            xArray.push(sample.in);
            yArray.push(sample.out);
        }

        const xs = tf.tensor2d(xArray, [batchSize, this.contextWindowSize]);
        const ys = tf.tensor2d(yArray, [batchSize, this.vocabSize]);

        const loss = await this.model.trainOnBatch(xs, ys);
        lossValue = Array.isArray(loss) ? loss[0] : loss;

        xs.dispose();
        ys.dispose();

        return lossValue;
    }

    decayLearningRate(factor = 0.999) {
        this.learningRate = this.learningRate * factor;
        // The optimizer stores the learning rate inside the dense class
        this.model.optimizer.learningRate = this.learningRate;
    }

    getRawState() {
        const weights = this.model.getWeights();

        let ihArr = [];
        let h1h2Arr = [];
        let hoArr = [];

        // Build array of arrays for the visualizer to avoid breaking ui.js
        if (weights.length >= 6) {
            const ih = weights[0].arraySync(); // [vocabSize, hiddenNodes]
            for (let i = 0; i < this.hiddenNodes; i++) {
                ihArr[i] = [];
                for (let j = 0; j < this.vocabSize; j++) {
                    ihArr[i][j] = ih[j][i];
                }
            }

            const h1h2 = weights[2].arraySync(); // [hiddenNodes, hiddenNodes]
            h1h2Arr = h1h2; // Already in correct format for rendering H to H

            const ho = weights[4].arraySync(); // [hiddenNodes, vocabSize]
            for (let i = 0; i < this.vocabSize; i++) {
                hoArr[i] = [];
                for (let j = 0; j < this.hiddenNodes; j++) {
                    hoArr[i][j] = ho[j][i];
                }
            }
        }

        const rawWeights = weights.map(w => ({
            data: Array.from(w.dataSync()),
            shape: w.shape
        }));

        return {
            vocabSize: this.vocabSize,
            hiddenNodes: this.hiddenNodes,
            weightsIH: ihArr,
            weightsH1H2: h1h2Arr,
            weightsHO: hoArr,
            weights: rawWeights
        };
    }

    setRawState(data) {
        if (data.vocabSize) this.vocabSize = data.vocabSize;
        if (data.hiddenNodes) this.hiddenNodes = data.hiddenNodes;

        if (!this.model) this.initModel();

        if (data.weights) {
            tf.tidy(() => {
                const tensors = data.weights.map(w => tf.tensor(w.data, w.shape));
                this.model.setWeights(tensors);
            });
        }
    }

    serialize() {
        return JSON.stringify(this.getRawState());
    }

    deserialize(dataStr) {
        const data = JSON.parse(dataStr);
        this.setRawState(data);
    }
}
