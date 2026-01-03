from transformers import MT5Model,MT5Config

config=MT5Config(d_model=256)
model = MT5Model.from_pretrained("google/mt5-small",config=config)

""" HF builds a 256 model but the checkpoints expects a 512 dim
    hence it causes a shape missmatch
    there for the architecture fields are non negotiable"""