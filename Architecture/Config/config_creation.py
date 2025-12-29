from transformers import AutoConfig,MT5Config,MT5Model

# methods for creating config object

# method1
# using the AutoConfig
# this the most used method 
# can be created using the model name or the local config.json
config1=AutoConfig.from_pretrained("google/mt5-small")
config2=AutoConfig.from_pretrained("./my_model_dir/")

# method2
# using the model's config class constructor
config3= MT5Config()


# method 3
# using the model's config Class  methods
# can be done in two ways either using config dictionary or config.json
config_dict={
    
}
config4= MT5Config.from_dict(config_dict)
config5=MT5Config.from_json_file("config.json")

# method 4
# creating config by cloning
old_config = AutoConfig.from_pretrained("google/mt5-small")
config6 = old_config.clone()

# method 5
# using the model's class 
# the returned model consist config in it
model = MT5Model.from_pretrained("google/mt5-small")
config7 =model.config