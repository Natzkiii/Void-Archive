from transformers import OPTForCausalLM, AutoTokenizer
import torch

device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
model = OPTForCausalLM('facebook/opt-350m')

model.load_state_dict(torch.load(''))

inputs = tokenizer.encode('<|system|>\n### Instructions\nThis is your character use these information and act according to this.\n\n## Character Information\nName: Void Archive.\nAge: 20.\nGender: Male.\nHeight: 182 CM.\nEye Color: Purple.\nHair Color: Golden Blond.\nSkin Color: Pale White.\nModel Type: Assistant.\nTags: Honkai Impact 3rd.\n\n# Clothing\nHe is wearing a coat with outer parts white and inner parts gray.\nHe Is wearing a black shirt underneath his coat and long gray pants with black shoes.\n\n## Personality.\nVoid Archive is emotionless but respectful.\nHe talk in formal English.\nHe will answer any questions asked by the user.\n\n<|user|>What is honkai.\n', return_tensors='pt').to(device)
attention_mask = torch.ones_like(inputs).to(device)
generate = model.generate(
                  inputs,
                  attention_mask=attention_mask,
                  max_new_tokens=512,
                  min_new_tokens=50,
                  do_sample=True,
                  num_beams=1,
                  top_p=0.95,
                  temperature=0.1,
                  repetition_penalty=1.0,
                  use_cache=True,
                  pad_token_id=tokenizer.eos_token_id,
                  early_stopping=False
                  )

output = tokenizer.decode(generate[:, inputs.shape[-1]:][0], skip_special_tokens=True)
print(output)