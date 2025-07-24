from pyarrow import output_stream
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# to load the pretrained checkpoint
model = AutoModel.from_pretrained("bert-base-cased")

# to save the config and models locally
model.save_pretrained("models/")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# converts from text to i/p ids
encoded_input = tokenizer(["How are you?","I'm fine, thank you!"],
                          padding=True,
                          return_tensors='pt')
print(encoded_input)

simulated_ip = "* " * 250
long_encoded_input = tokenizer([
    f"{simulated_ip} This is a very very very very very very very very very very very very very very very very very "
    f"very very very very very very very very very very very very very very very very very very very very very very "
    f"very very very very very very very very very very long sentence. {simulated_ip}",
    f"second big bing big sentence"],
    truncation=True,
    padding=True,
    max_length = 100,
    return_tensors = "pt"
)

output = model(**long_encoded_input)

# multiple sequences

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# sent to tokens
tokens = tokenizer.tokenize(sequence)

# text to numbers
ids = tokenizer.convert_tokens_to_ids(tokens)

# convert as tensors as models expects feeding that way
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

#check for logits
output = model(input_ids)
print("Logits:", output.logits)

