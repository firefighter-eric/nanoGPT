import json

from transformers import OPTForCausalLM, GPT2Model

from model import GPT


def save_model_detail(model, model_name):
    with open(f'data/models/{model_name}.txt', 'w') as f:
        # config
        f.write(json.dumps(model.config.__dict__, default=str, indent=2))
        f.write('\n')
        f.write('-' * 100)
        f.write('\n')

        # model
        f.write(str(model))
        f.write('\n')
        f.write('-' * 100)
        f.write('\n')

        # state dict keys
        f.write(
            str(json.dumps(list(dict(model.state_dict()).keys()), indent=2))
        )
        f.write('\n')
        f.write('-' * 100)
        f.write('\n')


# %% nano gpt
gpt_model_name = 'gpt2'
gpt = GPT.from_pretrained(gpt_model_name)
save_model_detail(gpt, gpt_model_name)

# %% gpt2
gpt2_model_path = 'gpt2'
gpt2_hf = GPT2Model.from_pretrained(gpt2_model_path)
save_model_detail(gpt2_hf, 'gpt2_hf')

# %% opt
opt_model_path = 'facebook/opt-125m'
opt = OPTForCausalLM.from_pretrained(opt_model_path)
save_model_detail(opt, 'opt-125m')
