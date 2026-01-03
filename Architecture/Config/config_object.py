from transformers import AutoConfig,AutoModelForSeq2SeqLM
import torch
import pprint
""" the creation method of the config does not matter, they all return the same object"""

config=AutoConfig.from_pretrained("google/mt5-small",torch_dtype= torch.float16)
print("Config class:",config.__class__)
print("Config values:",pprint.pprint(config.to_dict())) 
print("Total config fields:",len(config.to_dict()))


identity_fields = [
    "model_type",
    "architectures",
    "is_encoder_decoder",
    "dtype",
    "transformers_version",
    "initializer_factor"
]

print("\n=== IDENTITY & METADATA ===\n")
for f in identity_fields:
    print(f"{f}: {getattr(config, f, None)}")

architecture_fields = [
    "vocab_size",
    "d_model",
    "num_layers",
    "num_decoder_layers",
    "num_heads",
    "d_kv",
    "d_ff",
    "feed_forward_proj"
]

print("\n=== ARCHITECTURE SHAPE ===\n")
for f in architecture_fields:
    print(f"{f}: {getattr(config, f, None)}")

attention_fields = [
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "attention_dropout_rate",
    "layer_norm_epsilon"
]

print("\n=== ATTENTION MECHANICS ===\n")
for f in attention_fields:
    print(f"{f}: {getattr(config, f, None)}")

training_fields = [
    "dropout_rate",
    "classifier_dropout",
    "initializer_factor",
    "tie_word_embeddings",
    "use_cache"
]

print("\n=== TRAINING & REGULARIZATION ===\n")
for f in training_fields:
    print(f"{f}: {getattr(config, f, None)}")

runtime_fields = [
    "output_attentions",
    "output_hidden_states",
    "return_dict"
]

print("\n=== OUTPUT & RUNTIME CONTROL ===\n")
for f in runtime_fields:
    print(f"{f}: {getattr(config, f, None)}")

token_fields = [
    "pad_token_id",
    "eos_token_id",
    "decoder_start_token_id"
]

print("\n=== TOKEN IDS ===\n")
for f in token_fields:
    print(f"{f}: {getattr(config, f, None)}")


generation_fields = [
    "max_length",
    "min_length",
    "num_beams",
    "length_penalty",
    "early_stopping",
    "no_repeat_ngram_size"
]

print("\n=== GENERATION DEFAULTS ===\n")
for f in generation_fields:
    print(f"{f}: {getattr(config, f, None)}")
