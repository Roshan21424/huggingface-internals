from transformers import MT5Model

model = MT5Model.from_pretrained("google/mt5-small")
print(model.config.d_model)
