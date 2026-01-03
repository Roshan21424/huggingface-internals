from transformers import MT5Model,MT5Config

config = MT5Config(d_model=256)
model = MT5Model.from_pretrained("google/mt5-small",config=config,ignore_mismatched_sizes=True)

"""
- HF loads the compatable tensors and ignores the incompatible ones 
- and then reinitializes skipped layers
- the model loads but the outputs are garbage and no error
- this exists for head replacement, not architecture surgery.
"""
