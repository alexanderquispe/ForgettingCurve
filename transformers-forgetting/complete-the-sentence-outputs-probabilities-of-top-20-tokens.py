from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the model and tokenizer for T5
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Your text prompt
prompt = "The weather today is"

# Encode the text prompt to get the input IDs
input_ids = tokenizer(prompt, return_tensors='pt').input_ids

# Generate decoder_input_ids
decoder_start_token = model.config.decoder_start_token_id
decoder_input_ids = torch.full(
    (input_ids.size(0), 1),
    decoder_start_token,
    dtype=torch.long,
    device=input_ids.device
)

# Generate the output tokens with logits
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

# Get the logits for the first output token position
logits = outputs.logits[:, -1, :]

# Convert logits to probabilities
probabilities = torch.softmax(logits, dim=-1)

# Get the top 20 tokens and their probabilities
top_probs, top_indices = torch.topk(probabilities, 20)

# Convert token indices to actual tokens
tokens = [tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in top_indices[0]]

# Combine tokens and probabilities into tuples and store in a list
top_tokens_with_probs = list(zip(tokens, top_probs[0].tolist()))

# Print or use the top tokens with their probabilities
print(top_tokens_with_probs)
