const fs = require('fs').promises
const natural = require('natural');
const {
  WordTokenizer
} = natural;
const NeuralNetwork = require('./NeuralNetwork');

const nlp = require('compromise');


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


  capitalizeFirstLetterOfEachSentence(text) {
    let sentences = text.match(/[^\.!\?]+[\.!\?]+/g); // This splits the text into sentences
    if (sentences == null) { // If no sentence-ending punctuation was found, treat the whole text as a single sentence.
      sentences = [text];
    }
    sentences = sentences.map(sentence =>
      sentence.charAt(0).toUpperCase() + sentence.slice(1)
    );
    return sentences.join(" ");
  }

  correctGrammar(text) {
    let doc = nlp(text);

    doc.normalize({
      numbers: true, // Convert 'two' to 2
    });

    // Fix punctuation
    let sentences = doc.sentences().data();
    sentences.forEach((sentence) => {
      if (!/[.!?\n]$/.test(sentence.text)) {
        sentence.text += '.'; // Append a period to each sentence if it doesn't already end with a punctuation
      }
    });

    // Convert to a single paragraph
    let paragraph = sentences.map((sentence) => sentence.text).join(' ');

    paragraph = this.capitalizeFirstLetterOfEachSentence(paragraph);

    return paragraph;
  }

  async validateResponse(response) {
    try {
      // Check the response length
      if (response.length < 1) {
        return false;
      }

      // Check for punctuation
      if (response[response.length - 1] !== '.' && response[response.length - 1] !== '!') {
        return false;
      }

      // Load a pre-trained language model
      const languageModel = await tf.loadLayersModel('file://./trained_model/model.json');

      // Convert the response to a sequence of word indices
      let sequence = response.split(' ').map(word => this.uniqueWords.indexOf(word)).filter(index => index !== -1);

      // Pad or truncate the sequence to match the maximum sequence length
      let maxSequenceLength = 0;

      for (const data of this.trainingData) {
        const tokens = this.tokenizer.tokenize(data.input);
        const sequenceLength = tokens.length;

        if (sequenceLength > maxSequenceLength) {
          maxSequenceLength = sequenceLength;
        }
      }
      if (sequence.length > maxSequenceLength) {
        sequence = sequence.slice(0, maxSequenceLength);
      } else {
        sequence.push(...new Array(maxSequenceLength - sequence.length).fill(0));
      }

      // Convert the sequence to a 2D tensor
      const responseTensor = tf.tensor2d([sequence]);

      // Predict the next word in the response
      const prediction = languageModel.predict(responseTensor);

      // Get the predicted word index
      const predictedWordIndex = tf.argMax(prediction.flatten()).dataSync()[0];

      // Print the predicted word
      const predictedWord = this.uniqueWords[predictedWordIndex];
      console.log(predictedWord);

      return true; // Return true as the validation passed
    } catch (error) {
      console.error('Error validating response:', error);
      return false;
    }
  }





  paraphrase(response) {
    const tokens = this.tokenizer.tokenize(response);
    let attempts = 0;
    let paraphrasedResponse = "";

    while (attempts < this.maxRetries) {
      const newTokens = tokens.map((token, index) => {
        // Replace the token with a random synonym if available
        if (this.synonyms.has(token) && Math.random() < this.reward) {
          const synonyms = this.synonyms.get(token);
          return synonyms[Math.floor(Math.random() * synonyms.length)];
        }
        // Replace the token with a random pattern if available
        if (this.wordPatterns.has(token) && Math.random() < this.reward) {
          const patterns = Array.from(this.wordPatterns.get(token));
          const contextTokens = [
            tokens[index - 1] || "",
            token,
            tokens[index + 1] || ""
          ];
          const matchingPatterns = patterns.filter(pattern =>
            contextTokens.includes(pattern)
          );
          if (matchingPatterns.length > 0) {
            return matchingPatterns[Math.floor(Math.random() * matchingPatterns.length)];
          }
        }
        return token; // Return the original token if no replacement is made
      });

      paraphrasedResponse = newTokens.join(" ");
      if (!this.trainingData.some(data => data.output === paraphrasedResponse)) {
        break; // Break the loop if a unique response is found
      }
      attempts++;
    }

    return paraphrasedResponse;
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
    const maxInputLength = Math.max(...this.trainingData.map(({
      input
    }) => this.tokenizer.tokenize(input).length));

    // Add an embedding layer to convert word indices into dense vectors
    model.add(tf.layers.embedding({
      inputDim: vocabularySize,
      outputDim: 64,
      inputLength: maxInputLength, // Set the maximum input sequence length
    }));

    // Add an LSTM layer with 32 units (adjust as needed)
    model.add(tf.layers.lstm({
      units: 32
    }));

    // Add a dense (fully connected) layer for output
    model.add(tf.layers.dense({
      units: vocabularySize,
      activation: 'softmax'
    }));

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
      const tokenizedInput = this.trainingData.map(({
        input
      }) => this.tokenizer.tokenize(input));
      const tokenizedOutput = this.trainingData.map(({
        output
      }) => this.tokenizer.tokenize(output));

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
      model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy'
      });

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



  async loadTrainingDataFromFile(path) {
    try {
      const fileContents = await fs.readFile(path, 'utf-8');
      const trainingData = JSON.parse(fileContents);
      return trainingData;
    } catch (error) {
      console.error('Error loading training data:', error);
      return [];
    }
  }

  async loadTrainingData(filePath) {
    try {
      const trainingData = await this.loadTrainingDataFromFile(filePath);

      if (!Array.isArray(trainingData)) {
        throw new Error('Invalid training data format. Expected an array.');
      }

      this.trainingData = trainingData.map(({
        instruction,
        input,
        response
      }) => ({
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


  async loadTrainedModel() {
    try {
      const model = await tf.loadLayersModel('file://trained_model/model.json');
      this.trainedModel = model;
      console.log('Trained model loaded successfully.');
    } catch (error) {
      console.error('Error loading trained model:', error);
    }
  }

  async generateResponse(input, temperature) {
    try {
      const tokens = this.tokenizer.tokenize(input);
      await this.loadTrainedModel();

      let response = "";
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

        response = predictedWord;
      } else {
        response = this.classifier.classify(tokens);
      }

      // Paraphrase the response to ensure uniqueness
      let paraphrasedResponse = this.paraphrase(response);
      let correctedResponse = this.correctGrammar(paraphrasedResponse);

      return correctedResponse;
    } catch (error) {
      console.error('Error generating unique response:', error);
      return null; // Return null or a default response in case of an error
    }
  }

   async generateUniqueResponse(input, attempts = 0) {
    try {
      if (attempts > this.maxRetries) {
        console.log('Max retries exceeded. Returning best attempt.');
        return this.generateResponse(input);
      }

      let response = await this.generateResponse(input);
      
      if (this.trainingData.some(data => data.output === response)) {
        console.log('Response already exists in the training data. Retrying.');
        return await this.generateUniqueResponse(input, attempts + 1);
      }

      let isValid = await this.validateResponse(response);
      
      if (!isValid) {
        console.log('Invalid response. Retrying.');
        return await this.generateUniqueResponse(input, attempts + 1);
      }

      return response;
    } catch (error) {
      console.error('Error generating unique response:', error);
      return null; // Return null or a default response in case of an error
    }
  }

}

module.exports = ChatBot;
