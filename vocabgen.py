import json
from tokenizers import ByteLevelBPETokenizer

# Define paths to input text file, output vocab file, and output merges file
input_file = "vocab_info.txt"
vocab_file = "vocab.json"
merges_file = "merges.txt"
# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()

# Define special tokens
special_tokens = ["<|unk|>", "<|pad|>", "<|endoftext|>", "<|sep|>", "<|mask|>", "<|beginningoftext|>"]
#non_define_tokens = ["vinuk", "pasindu"]
#tokenizer.add_tokens(non_define_tokens)

# Train tokenizer on input text file
tokenizer.train(files=[input_file], vocab_size=100000, special_tokens=special_tokens)

# Save vocabulary to JSON file
with open(vocab_file, "w") as outfile:
    json.dump(tokenizer.get_vocab(), outfile)

# Save merges to text file
tokenizer.save_model(".", "my_tokenizer")
with open(merges_file, "w") as outfile:
    outfile.write(open("my_tokenizer-merges.txt", "r").read())
