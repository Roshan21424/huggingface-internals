from transformers import MT5Model,AutoModelForSeq2SeqLM
""" the are two types of models in HF: base model and task model"""

# base model
base_model = MT5Model.from_pretrained("google/mt5-small")
""" Core Transformer architecture only
- they include embeddings, encoder, decoder, attention, FFNs etc
"""

# task model
task_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
""" Core Transformer architecture only
- they include embeddings, encoder, decoder, attention, FFNs etc
- also contains final head
"""


""" weight loading in HF is done like:
- it loads weights only parameter name and parameter shape
- it does not check semantic meaning
- task correctness
- output quality
"""


""" we cannot change architectural fields in transformer
- example d_model, num_layers, num_heads, d_ff, vocab_size
- these define tensor shapes, if shape mismatch then it will cause a hard failure
- we can only change the behavioral fields like use_cache, output_hidden_states, return_dict and so on as they do not affect weights or shapes
"""


""" heads are replaceable 
- we can change task heads, number of heads, and sometimes head dimensions
- HF is designed keep backbone and swap heads
"""


""" tied weights amplify breakage
- example for MT5
shared.weight
encoder.embed_tokens.weight
decoder.embed_tokens.weight
lm_head.weight
- all of these point to same underlying tensor
- if we change something like vocab_size and embedding dimensions then many
layers may break at once
- This change would require coordinated surgery across tied tensors,HF simply rejects to do so
"""


""" how to identify base model and task model
base model: 
- class name generally ends with model, same patterns also follows for Automodel
- output contains hidden states and attentions
- contains encoder and decode

task model:
- class name generally contains for, same patterns also follows for Automodel
- contains logits and loss
- contains lm_head and classifier

print(hasattr(MT5Model.from_pretrained("google/mt5-small"), "lm_head"))
print(hasattr(AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small"), "lm_head"))

- model.config.architectures tell us which task model HF expects base model does not have architectures
"""