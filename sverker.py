# import the discord library
import discord

#import the torch library
import torch

#import the transformers library
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria

#import the os library
import os

# Create a new Intents object with default settings
intents = discord.Intents.default()

# Enable the bot to receive updates about message events
intents.messages = True

# Enable the bot to receive updates about guild (server) events
intents.guilds = True

# Create a new Client object for the bot, specifying the events it should receive updates about
client = discord.Client(intents=intents)

# selcet the model to use
model_name = "AI-Sweden-Models/gpt-sw3-126m-instruct"

# in case you want to use a GPU, otherwise it will use the CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize Tokenizer & Model

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a pre-trained causal language model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Put the model into evaluation mode
model.eval()

# Move the model to the specified device (CPU or GPU)
model.to(device)

# This decorator registers an event for the bot
@client.event

# This asynchronous function is triggered when the bot is ready
async def on_ready():
    # Print a message to the console indicating that the bot has logged in, for debugging
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    # Don't respond to ourselves
    if message.author == client.user:
        return

    # Check if the bot is mentioned
    if client.user.mentioned_in(message):
        #get the bot name
        bot_name = client.user.name
        
        #get the message and remove the bot name
        message_clean = message.content.replace(f'<@{client.user.id}>', bot_name).strip()
        
        #print the message for debugging
        print(message_clean)
        
        # Create a prompt for the language model
        prompt = f'<|endoftext|> <s> User: {message_clean} <s> Bot: '.strip()
        
        #print the prompt for debugging
        print(prompt)

        # Define a custom stopping criteria for the language model
        class StopOnTokenCriteria(StoppingCriteria):
            # Initialize the class with the ID of the stop token
            def __init__(self, stop_token_id):
                self.stop_token_id = stop_token_id

            # This method is called to check if the model should stop generating text
            def __call__(self, input_ids, scores, **kwargs):
                # Check if the last token generated is the stop token
                return input_ids[0, -1] == self.stop_token_id

        # Create a stopping criteria that stops text generation when the BOS token is encountered
        stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=tokenizer.bos_token_id)

        # Tokenize the prompt, convert it into input IDs, and move the tensors to the specified device
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Generate text using the language model
        generated_token_ids = model.generate(
            # Specify the input IDs
            inputs=input_ids,
            # Specify the maximum number of tokens to generate
            max_new_tokens=400,
            # do_sample means that the model will use sampling to select next token instead of the greedy argmax approach
            do_sample=True,
            # temperature controls the randomness of the sampling, lower temperature results in less randomness
            temperature=0.4,
            #top_p controls diversity via nucleus sampling, 1.0 means no restrictions
            top_p=1,
            #repetition_penalty controls how much the model avoids repeating itself, 1.0 means no penalty
            repetition_penalty=1.1,
            #stop_token is the token ID at which text generation is stopped
            stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
        )[0]

        # Decode the generated text
        generated_text = tokenizer.decode(generated_token_ids[len(input_ids[0]):-1])     
        
        # Print the generated text for debugging
        print(generated_text)
        
        # Send the response back to Discord as a reply
        await message.reply(generated_text)

# Start the Discord bot, sverker_token is the bot token stored as an environment variable that was set in the terminal using setx SVERKER_TOKEN <token>
client.run(os.environ.get('SVERKER_TOKEN')) 