const { 
    Client, 
    IntentsBitField, 
    ActivityType,
    SlashCommandBuilder 
} = require('discord.js');


const { DISCORD_TOKEN } = require('./config');

const client = new Client({
  intents: [
    IntentsBitField.Flags.Guilds,
    IntentsBitField.Flags.GuildMembers,
    IntentsBitField.Flags.GuildMessages,
    IntentsBitField.Flags.MessageContent
  ]
});

const commands = [
  new SlashCommandBuilder()
    .setName('chat')
    .setDescription('Neuron is an AI model which you can talk to.')
    .addStringOption(option =>
      option.setName('message').setRequired(true)
        .setDescription('message for the AI to give a response back.')),
];

client.on('ready', () => {

  client.application.commands.set(commands)
    .then(() => console.log('Slash commands registered successfully!'))
    .catch(console.error);

  client.user.setActivity({
    name: '/chat to talk to me',
    type: ActivityType.Playing
  });
});


client.login(DISCORD_TOKEN);

module.exports = client