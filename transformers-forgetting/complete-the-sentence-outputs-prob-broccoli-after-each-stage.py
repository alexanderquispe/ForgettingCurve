from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the model and tokenizer for T5
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Your text prompt
#prompt = "I always buy on-line from this store. The idea of becoming a prime member is "
prompt = "I was thinking of product, wood, new, walk, not, and"
print(prompt)

# Encode the text prompt to get the input IDs
input_ids = tokenizer(prompt, return_tensors='pt').input_ids

# Token ID for 'cauliflower'
cauliflower_token_id = tokenizer.encode('a', add_special_tokens=False)[0]
print(cauliflower_token_id)

# Initialize decoder_input_ids
decoder_input_ids = torch.full(
    (1, 1),
    model.config.decoder_start_token_id,
    dtype=torch.long
)

# Store the probabilities of 'cauliflower' at each stage
cauliflower_probabilities = []

# Loop to generate each token and get the probability of 'cauliflower'
for step in range(50):  # assuming a maximum length of 50 tokens
    # Generate the output tokens with logits up to the current step
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
    logits = outputs.logits

    # Get the logits for the last token position
    next_token_logits = logits[:, -1, :]

    # Convert logits to probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the probability of 'cauliflower'
    cauliflower_probability = probabilities[:, cauliflower_token_id].item()
    cauliflower_probabilities.append(cauliflower_probability)

    # Print the probability of 'cauliflower'
    print(f"Step {step}: Probability of 'a': {cauliflower_probability}")

    # Generate the next token ID (greedy)
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    # Check if the next token is the end of sequence token
    if next_token_id == model.config.eos_token_id:
        break

    # Append the next token ID to the decoder_input_ids for the next iteration
    decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

# Print the final generated sequence
final_sequence = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
print(f"Final generated sequence: {final_sequence}")
