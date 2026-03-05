import * as tf from 'https://esm.sh/@tensorflow/tfjs@4.22.0';

export class AdvancedBrain {
    constructor(vocabSize, hiddenNodes) {
        this.vocabSize = vocabSize;
        this.hiddenNodes = hiddenNodes;
        this.learningRate = 0.5; // Bumped slightly for TFJS SGD
        this.momentum = 0.9;
        this.initModel();
    }

    initModel() {
        this.model = tf.sequential();

        // Input: vocabSize -> Hidden 1: hiddenNodes (ReLU)
        this.model.add(tf.layers.dense({
            units: this.hiddenNodes,
            inputShape: [this.vocabSize],
            activation: 'relu',
            kernelInitializer: 'heNormal',
            useBias: true
        }));

        // Hidden 1: hiddenNodes -> Hidden 2: hiddenNodes (ReLU)
        this.model.add(tf.layers.dense({
            units: this.hiddenNodes,
            activation: 'relu',
            kernelInitializer: 'heNormal',
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
            optimizer: tf.train.momentum(this.learningRate, this.momentum),
            loss: 'categoricalCrossentropy'
        });
    }

    expandVocab(newSize) {
        if (newSize <= this.vocabSize) return;

        const oldWeights = this.model.getWeights();
        // Index 0: IH Kernel [oldVocabSize, hiddenNodes]
        // Index 1: IH Bias   [hiddenNodes]
        // Index 2: H1H2 Kernel [hiddenNodes, hiddenNodes]
        // Index 3: H1H2 Bias   [hiddenNodes]
        // Index 4: HO Kernel [hiddenNodes, oldVocabSize]
        // Index 5: HO Bias   [oldVocabSize]

        const newModel = tf.sequential();
        newModel.add(tf.layers.dense({
            units: this.hiddenNodes,
            inputShape: [newSize],
            activation: 'relu',
            kernelInitializer: 'heNormal',
            useBias: true
        }));
        newModel.add(tf.layers.dense({
            units: this.hiddenNodes,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            useBias: true
        }));
        newModel.add(tf.layers.dense({
            units: newSize,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal',
            useBias: true
        }));

        newModel.compile({
            optimizer: tf.train.momentum(this.learningRate, this.momentum),
            loss: 'categoricalCrossentropy'
        });

        tf.tidy(() => {
            // IH Kernel Pad
            const ihKernelPad = tf.randomNormal([newSize - this.vocabSize, this.hiddenNodes], 0, Math.sqrt(2 / newSize));
            const newIhKernel = tf.concat([oldWeights[0], ihKernelPad], 0);

            // IH Bias is unchanged in shape
            const newIhBias = oldWeights[1].clone();

            // H1H2 Kernel & Bias are completely untouched
            const newH1H2Kernel = oldWeights[2].clone();
            const newH1H2Bias = oldWeights[3].clone();

            // HO Kernel Pad (Now Index 4)
            const hoKernelPad = tf.randomNormal([this.hiddenNodes, newSize - this.vocabSize], 0, Math.sqrt(1 / this.hiddenNodes));
            const newHoKernel = tf.concat([oldWeights[4], hoKernelPad], 1);

            // HO Bias Pad (Now Index 5)
            const hoBiasPad = tf.zeros([newSize - this.vocabSize]);
            const newHoBias = tf.concat([oldWeights[5], hoBiasPad], 0);

            newModel.setWeights([newIhKernel, newIhBias, newH1H2Kernel, newH1H2Bias, newHoKernel, newHoBias]);
        });

        this.model.dispose(); // Releases old tensors
        this.model = newModel;
        this.vocabSize = newSize;
    }

    predict(inputVector) {
        return tf.tidy(() => {
            const inputTensor = tf.tensor2d([inputVector], [1, this.vocabSize]);
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

        const xs = tf.tensor2d(xArray, [batchSize, this.vocabSize]);
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
