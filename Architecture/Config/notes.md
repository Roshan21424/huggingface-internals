key takeaways — hugging face config introspection

1. config creation method does not matter.
   autoconfig, model-specific config, json, dict, clone — all produce the same
   pretrainedconfig object once instantiated. the origin is irrelevant afterward.

2. a config is the single source of truth for model architecture.
   fields like d_model, num_layers, num_heads, d_ff, vocab_size define tensor
   shapes. changing any of these breaks compatibility with pretrained weights.

3. config fields fall into clear functional categories:
   - identity & metadata: model_type, architectures, transformers_version
      used for automodel dispatch, serialization, and compatibility.
   - architecture shape: d_model, num_layers, num_heads, d_ff
      hard constraints; must match weights exactly.
   - attention mechanics: relative_attention_* fields
      define positional bias behavior (t5/mt5-specific).
   - training & regularization: dropout_rate, tie_word_embeddings, initializer_factor
      affect training dynamics, safe to tweak post-load.
   - runtime/output flags: output_hidden_states, return_dict
     do not affect model parameters, only outputs.
   - token ids: pad_token_id, eos_token_id, decoder_start_token_id
      must align with tokenizer or silent bugs occur.
   - generation defaults: num_beams, max_length, length_penalty
      policy defaults for generation, not part of core model architecture.

4. config declares intent; it does not enforce runtime behavior.
   example: torch_dtype in config expresses preferred dtype, but actual dtype
   is enforced during model loading or via .to() / autocast.

5. autoconfig is a factory, not a config type.
   it reads model_type from config.json and returns the correct subclass
   (e.g., mt5config). after creation, only the returned object matters.

6. hugging face loading order is always:
   config → model skeleton → weights (if available)
   architecture is defined before weights are applied, never the reverse.

7. generation behavior is partly config-driven but overrideable.
   generation parameters in config act as defaults and can be overridden at
   call time (model.generate(...)).

mental model:
config = architectural + behavioral contract between model code, weights,
tokenizer, runtime, and generation logic.
