const { WordTokenizer } = require('natural');

class NeuralNetwork {
  constructor(hiddenSizes, outputSize) {
    this.hiddenSizes = hiddenSizes;
    this.outputSize = outputSize;
    this.weights = [];
    this.biases = [];
    this.threshold = 0.6;
    this.tokenizer = new WordTokenizer();
  }

  initializeWeightsAndBiases(inputSize) {
    const layerSizes = [inputSize, ...this.hiddenSizes, this.outputSize];

    for (let i = 1; i < layerSizes.length; i++) {
      const prevLayerSize = layerSizes[i - 1];
      const currentLayerSize = layerSizes[i];

      const layerWeights = [];
      const layerBiases = [];
      for (let j = 0; j < currentLayerSize; j++) {
        const weights = [];
        for (let k = 0; k < prevLayerSize; k++) {
          weights.push(Math.random() - 0.5); // Random weight between -0.5 and 0.5
        }
        layerWeights.push(weights);
      }
      this.weights.push(layerWeights);

      const biases = [];
      for (let j = 0; j < currentLayerSize; j++) {
        biases.push(Math.random() - 0.5); // Random bias between -0.5 and 0.5
      }
      this.biases.push(biases);
    }
  }

  loadWeightsAndBiases(weights, biases) {
    this.weights = weights;
    this.biases = biases;
  }

  sigmoid(x) {
    const threshold = 15; // Adjust the threshold as needed
    if (x > threshold) {
      return 1;
    } else if (x < -threshold) {
      return 0;
    }
    return 1 / (1 + Math.exp(-x));
  }

  feedForward(input) {
    let activation = input;
    for (let i = 0; i < this.hiddenSizes.length; i++) {
      const layerWeights = this.weights[i];
      const layerBiases = this.biases[i];

      const layerOutput = [];
      for (let j = 0; j < layerWeights.length; j++) {
        let sum = layerBiases[j];
        for (let k = 0; k < layerWeights[j].length; k++) {
          sum += layerWeights[j][k] * activation[k];
        }
        const output = this.sigmoid(sum);
        layerOutput.push(output);
      }
      activation = layerOutput.map((value) => (isNaN(value) || !isFinite(value) ? 0 : value));
    }

    const outputLayerWeights = this.weights[this.weights.length - 1];
    const outputLayerBiases = this.biases[this.biases.length - 1];
    const output = [];
    for (let i = 0; i < outputLayerWeights.length; i++) {
      let sum = outputLayerBiases[i];
      for (let j = 0; j < outputLayerWeights[i].length; j++) {
        sum += outputLayerWeights[i][j] * activation[j];
      }
      const outputValue = this.sigmoid(sum);
      output.push(isNaN(outputValue) || !isFinite(outputValue) ? 0 : outputValue);
    }

    return output;
  }

  backpropagation(input, target, learningRate) {
    const activations = [input];
    const layerOutputs = [];

    let activation = input;
    for (let i = 0; i < this.hiddenSizes.length; i++) {
      const layerWeights = this.weights[i];
      const layerBiases = this.biases[i];

      const layerOutput = [];
      for (let j = 0; j < layerWeights.length; j++) {
        let sum = layerBiases[j];
        for (let k = 0; k < layerWeights[j].length; k++) {
          sum += layerWeights[j][k] * activation[k];
        }
        const output = this.sigmoid(sum);
        layerOutput.push(output);
      }
      activation = layerOutput;
      activations.push(activation);
      layerOutputs.push(layerOutput);
    }

    const outputLayerWeights = this.weights[this.weights.length - 1];
    const outputLayerBiases = this.biases[this.biases.length - 1];
    const output = [];
    for (let i = 0; i < outputLayerWeights.length; i++) {
      let sum = outputLayerBiases[i];
      for (let j = 0; j < outputLayerWeights[i].length; j++) {
        sum += outputLayerWeights[i][j] * activation[j];
      }
      const outputValue = this.sigmoid(sum);
      output.push(outputValue);
    }
    activations.push(output);

    const outputErrors = [];
    for (let i = 0; i < target.length; i++) {
      const outputError = target[i] - output[i];
      outputErrors.push(outputError);
    }

    let error = outputErrors;
    for (let i = this.hiddenSizes.length - 1; i >= 0; i--) {
      const layerOutput = layerOutputs[i];
      const nextLayerOutput = i === this.hiddenSizes.length - 1 ? output : layerOutputs[i + 1];

      const layerError = error;
      const layerWeights = this.weights[i];
      const layerBiases = this.biases[i];

      const nextLayerError = [];
      for (let j = 0; j < layerOutput.length; j++) {
        const delta = layerError[j] * layerOutput[j] * (1 - layerOutput[j]);
        for (let k = 0; k < layerWeights[j].length; k++) {
          layerWeights[j][k] += learningRate * delta * nextLayerOutput[k];
        }
        layerBiases[j] += learningRate * delta;
        let sum = 0;
        for (let k = 0; k < layerWeights[j].length; k++) {
          sum += layerWeights[j][k] * layerError[j] * layerOutput[j] * (1 - layerOutput[j]);
        }
        nextLayerError.push(sum);
      }
      error = nextLayerError;
    }
  }

  calculateError(target, predicted) {
    let error = 0;
    let validCount = 0;
    for (let i = 0; i < target.length; i++) {
      if (!isNaN(predicted[i]) && isFinite(predicted[i])) {
        error += Math.abs(target[i] - predicted[i]);
        validCount++;
      }
    }
    return validCount > 0 ? error / validCount : 0;
  }

     generateResponse(input, temperature) {
  const inputVector = this.convertInputToVector(input);
  const responseVector = this.feedForward(inputVector);

  // Apply temperature to control the randomness of the response
  const scaledResponseVector = responseVector.map((probability) => Math.pow(probability, 1 / temperature));
  const sumProbabilities = scaledResponseVector.reduce((acc, probability) => acc + probability, 0);

  // Normalize the probabilities to sum up to 1
  const normalizedResponseVector = scaledResponseVector.map((probability) => probability / sumProbabilities);

  // Generate a random value within [0, 1)
  const randomValue = Math.random();

  // Choose the response token based on the probabilities
  let accumulatedProbability = 0;
  for (let i = 0; i < normalizedResponseVector.length; i++) {
    accumulatedProbability += normalizedResponseVector[i];
    if (randomValue <= accumulatedProbability) {
      return [this.tokenizer.tokenizer_list[i]]; // Wrap the chosen token in an array
    }
  }

  // If for some reason, no token is chosen, return an empty array
  return "no response";
}
  convertVectorToTokens(vector) {
    const threshold = 0.5; // Adjust the threshold as needed
    const tokens = [];

    for (let i = 0; i < vector.length; i++) {
      const value = vector[i];
      if (value >= threshold) {
        tokens.push(this.tokenizer.tokenizer_list[i]);
      }
    }

    return tokens;
  }

  convertTokensToString(tokens) {
    return tokens.join(' ');
  }

  trainModel(trainingData, epochs, learningRate) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalError = 0;
      for (let i = 0; i < trainingData.length; i++) {
        const { input, output } = trainingData[i];

        const inputVector = this.convertInputToVector(input);
        const outputVector = this.convertOutputToVector(output);

        const predicted = this.feedForward(inputVector);
        const error = this.calculateError(outputVector, predicted);
        totalError += error;

        this.backpropagation(inputVector, outputVector, learningRate);
      }
      const averageError = totalError / trainingData.length;
    }
  }

  calculateSimilarity(vectorA, vectorB) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      magnitudeA += vectorA[i] * vectorA[i];
      magnitudeB += vectorB[i] * vectorB[i];
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  getUniqueResponseTokens(responseTokens, inputTokens, uniqueWords) {
    const uniqueResponseTokens = [];
    for (const token of responseTokens) {
      if (inputTokens.includes(token) || !uniqueWords.includes(token)) {
        const uniqueToken = this.getRandomWord(uniqueWords);
        uniqueResponseTokens.push(uniqueToken);
      } else {
        uniqueResponseTokens.push(token);
      }
    }
    return uniqueResponseTokens;
  }

  getUniqueWords(trainingData) {
    const uniqueWords = new Set();
    for (const data of trainingData) {
      const tokens = this.tokenizer.tokenize(data.input);
      for (const token of tokens) {
        uniqueWords.add(token);
      }
    }
    return Array.from(uniqueWords);
  }

  getRandomWord(uniqueWords) {
    const randomIndex = Math.floor(Math.random() * uniqueWords.length);
    return uniqueWords[randomIndex];
  }

  convertInputToVector(input) {
    const inputString = String(input);
    const inputVector = this.tokenizer.tokenize(inputString);
    return inputVector;
  }

    convertOutputToVector(output) {
    if (typeof output !== 'string') {
      output = String(output);
    }

    const words = output.toLowerCase().split(' '); // Split the output into words
    const outputVector = [];

    for (const word of words) {
      if (word.length > 0) {
        const charCodes = Array.from(word).map((char) => char.charCodeAt(0) / 255);
        outputVector.push(...charCodes);
        // Add a space (representing word boundary) at the end of each word except the last one
        if (word !== words[words.length - 1]) {
          outputVector.push(0);
        }
      }
    }
    return outputVector;
  }
}


module.exports = NeuralNetwork;
