const ChatBot = require('./ChatBot');
const DiscordBot = require('./DiscordBot');

const chatBot = new ChatBot();

const trainingDataURL = 'https://raw.githubusercontent.com/saintyboy1/NeuralBot/main/training.json';

async function trainChatBot() {
  try {
    await chatBot.loadTrainingData(trainingDataURL);
    await chatBot.trainLanguageModel(); // Use the new method to train the language model
    console.log('Chatbot is trained and ready!');
  } catch (error) {
    console.error('Error setting up the chatbot:', error);
  }
}

async function generateResponse() {
  try {
    // Now the bot is trained and ready to generate responses
    // You can call the generateUniqueResponse method with user input to get a response
    const userInput = 'Imagine you are an ambassador from planet Zogaria write a speech introducing your planet and the planets culture to Earth leaders.';
    const temperature = 0.8;
    const response = await chatBot.generateUniqueResponse(userInput, temperature); // Wait for the response
    console.log(response);
  } catch (error) {
    console.error('Error generating response:', error);
  }
}

// First, train the chatbot
trainChatBot().then(() => {
  // After training is complete, generate responses
  generateResponse();
});
