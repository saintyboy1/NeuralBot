const fs = require('fs');
const natural = require('natural');
const { WordTokenizer } = natural;
const NeuralNetwork = require('./NeuralNetwork');


const tf = require('@tensorflow/tfjs-node')

class ChatBot {
  constructor() {
    this.trainingData = [];
    this.tokenizer = new WordTokenizer();
    this.classifier = new natural.BayesClassifier();
    this.neuralNetwork = null;
    this.uniqueWords = ['hello', 'world', 'how']; // Add your demo words here
    this.synonyms = new Map();
    this.wordPatterns = new Map();
    this.maxRetries = 5; // Maximum number of retries for generating a valid response
    this.reward = 0.1; // Reward value for successful completion
  }


 getUniqueWordsFromTokens(...tokensArrays) {
    const uniqueWords = new Set();
    for (const tokens of tokensArrays) {
      for (const tokenArray of tokens) {
        for (const token of tokenArray) {
          uniqueWords.add(token);
        }
      }
    }
    return Array.from(uniqueWords);
  }

convertTokensToVectors(tokensArray, uniqueWords) {
  const vectors = [];

  for (const tokens of tokensArray) {
    const vector = tokens.map((token) => {
      const index = Array.from(uniqueWords).indexOf(token);
      return index === -1 ? 0 : index;
    });
    vectors.push(vector);
  }

  return vectors;
}




   
  createLanguageModel(vocabularySize) {
    const model = tf.sequential();

    // Find the maximum input sequence length
    const maxInputLength = Math.max(...this.trainingData.map(({ input }) => this.tokenizer.tokenize(input).length));

    // Add an embedding layer to convert word indices into dense vectors
    model.add(tf.layers.embedding({
      inputDim: vocabularySize,
      outputDim: 64,
      inputLength: maxInputLength, // Set the maximum input sequence length
    }));

    // Add an LSTM layer with 32 units (adjust as needed)
    model.add(tf.layers.lstm({ units: 32 }));

    // Add a dense (fully connected) layer for output
    model.add(tf.layers.dense({ units: vocabularySize, activation: 'softmax' }));

    return model;
  }

   padSequences(sequences, maxSeqLength, paddingValue) {
    return sequences.map(seq => {
      if (seq.length >= maxSeqLength) {
        return seq.slice(0, maxSeqLength);
      } else {
        const padding = new Array(maxSeqLength - seq.length).fill(paddingValue);
        return [...seq, ...padding];
      }
    });
  }
convertTokensToClassIndices(tokensArray, uniqueWords) {
    const classIndices = [];

    for (const tokens of tokensArray) {
      const classIndex = tokens.map((token) => {
        return Array.from(uniqueWords).indexOf(token);
      });
      classIndices.push(classIndex); // Update to include all class indices for each example
    }

    return classIndices;
  }
 async trainLanguageModel() {
  try {
    if (this.trainingData.length === 0) {
      console.error('No training data found. Please load training data first.');
      return;
    }

    // Preprocess the training data and convert it into sequences of numbers (tokens)
    const tokenizedInput = this.trainingData.map(({ input }) => this.tokenizer.tokenize(input));
    const tokenizedOutput = this.trainingData.map(({ output }) => this.tokenizer.tokenize(output));

    // Create a vocabulary of unique words from the training data
    const uniqueWords = new Set([...this.uniqueWords, ...this.getUniqueWordsFromTokens(tokenizedInput, tokenizedOutput)]);

    const inputVectors = this.convertTokensToVectors(tokenizedInput, uniqueWords);
    const outputVectors = this.convertTokensToClassIndices(tokenizedOutput, uniqueWords);

    // Find the maximum sequence lengths for both input and output
    const maxInputLength = Math.max(...inputVectors.map(seq => seq.length));
    const maxOutputLength = Math.max(...outputVectors.map(seq => seq.length));

    // Pad the input and output sequences to have the same length
    const paddedInputVectors = this.padSequences(inputVectors, maxInputLength, 0);
    const paddedOutputVectors = this.padSequences(outputVectors, maxOutputLength, 0);

    // Convert the input and output vectors to tensors
    const inputTensors = tf.tensor2d(paddedInputVectors);
    const outputTensors = tf.tensor2d(paddedOutputVectors);

    // Transpose the output tensors to have a shape of [num_examples, 1]
    const targetTensors = tf.tensor2d(paddedOutputVectors.map(seq => seq.slice(0, 1)));

    // Define the language model architecture
    const model = this.createLanguageModel(uniqueWords.size);

    // Compile the model with an optimizer and loss function
    model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy' });

    // Train the model on the data
    await model.fit(inputTensors, targetTensors, {
      epochs: 100,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onTrainEnd: async () => {
          // Save the trained model as a file
          const modelSavePath = 'file://./trained_model';
          await model.save(modelSavePath);

          console.log('Language model trained and saved successfully.');
        },
      },
    });
  } catch (error) {
    console.error('Error training language model:', error);
  }
}





  trainModel(epochs, learningRate) {
    if (this.trainingData.length === 0) {
      console.error('No training data found. Please load training data first.');
      return;
    }

    this.neuralNetwork = new NeuralNetwork([64, 32], 1);
    this.neuralNetwork.initializeWeightsAndBiases(this.neuralNetwork.convertInputToVector(''));
    const convertedTrainingData = this.trainingData.map(({ input, output }) => ({
      input: this.neuralNetwork.convertInputToVector(input),
      output: this.neuralNetwork.convertOutputToVector(output),
    }));
    this.neuralNetwork.trainModel(convertedTrainingData, epochs, learningRate);
    this.extractWordPatterns();
  }

  async loadTrainingDataFromURL(url) {
    try {
      const fetch = await import('node-fetch');
      const response = await fetch.default(url);
      if (!response.ok) {
        throw new Error(`Error loading training data: ${response.status} ${response.statusText}`);
      }
      const trainingData = await response.json();
      return trainingData;
    } catch (error) {
      console.error('Error loading training data:', error);
      return [];
    }
  }

  async loadTrainingData(url) {
    try {
      const trainingData = await this.loadTrainingDataFromURL(url);

      if (!Array.isArray(trainingData)) {
        throw new Error('Invalid training data format. Expected an array.');
      }

      this.trainingData = trainingData.map(({ instruction, input, response }) => ({
        input: `${instruction} ${input}`,
        output: response,
      }));

      // Train NLP classifier with training data
      for (const data of this.trainingData) {
        const tokens = this.tokenizer.tokenize(data.input);
        this.classifier.addDocument(tokens, data.output);

        // Extract unique words
        const uniqueTokens = [...new Set(tokens)];
        for (const token of uniqueTokens) {
          if (!this.uniqueWords.includes(token)) {
            this.uniqueWords.push(token);
          }
        }

        // Add synonyms to the map
        const inputTokens = this.tokenizer.tokenize(data.input);
        const outputTokens = this.tokenizer.tokenize(data.output);
        for (let i = 0; i < inputTokens.length; i++) {
          const inputToken = inputTokens[i];
          const outputToken = outputTokens[i];
          if (inputToken !== outputToken) {
            if (!this.synonyms.has(inputToken)) {
              this.synonyms.set(inputToken, []);
            }
            this.synonyms.get(inputToken).push(outputToken);
          }
        }
      }
      this.classifier.train();

      console.log('Training data loaded successfully.');
    } catch (error) {
      console.error('Error loading training data:', error);
    }
  }

  extractWordPatterns() {
    for (const data of this.trainingData) {
      const inputTokens = this.tokenizer.tokenize(data.input);
      const outputTokens = this.tokenizer.tokenize(data.output);

      for (let i = 0; i < inputTokens.length; i++) {
        const inputToken = inputTokens[i];
        const outputToken = outputTokens[i];

        if (inputToken !== outputToken) {
          if (!this.wordPatterns.has(inputToken)) {
            this.wordPatterns.set(inputToken, new Set());
          }

          this.wordPatterns.get(inputToken).add(outputToken);
        }
      }
    }
  }

    async loadTrainedModel() {
    try {
      const model = await tf.loadLayersModel('file://trained_model/model.json');
      this.trainedModel = model;
      console.log('Trained model loaded successfully.');
    } catch (error) {
      console.error('Error loading trained model:', error);
    }
  }
async generateUniqueResponse(input, temperature) {
  try {
    const tokens = this.tokenizer.tokenize(input);
    await this.loadTrainedModel();

    if (this.languageModel) {
      const inputVector = this.convertTokensToVectors([tokens], this.uniqueWords);

      // Find the maximum input sequence length
      const maxInputLength = Math.max(...inputVector.map(seq => seq.length));

      // Pad the input sequences to have the same length
      const paddedInputVectors = this.padSequences(inputVector, maxInputLength, 0);

      // Convert the input vectors to a tensor
      const inputTensor = tf.tensor2d(paddedInputVectors);

      // Reshape the tensor to match the expected shape [null, maxInputLength]
      const reshapedInputTensor = inputTensor.reshape([inputTensor.shape[0], inputTensor.shape[1]]);

      const prediction = this.languageModel.predict(reshapedInputTensor);

      // Get the predicted index from the tensor
      const predictedIndex = tf.argMax(prediction, 1).dataSync()[0];
      const predictedWord = Array.from(this.uniqueWords)[predictedIndex];

      return predictedWord;
    } else {
      return this.classifier.classify(tokens);
    }
  } catch (error) {
    console.error('Error generating unique response:', error);
    return null; // Return null or a default response in case of an error
  }
}








}

module.exports = ChatBot;
