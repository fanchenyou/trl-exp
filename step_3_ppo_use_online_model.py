from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, pipeline, set_seed

from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead 
from trl.trainer.ppo_trainer import PPOConfig, PPOTrainer
from trl.core import LengthSampler

# step-3 performs PPO training
# https://github.com/lvwerra/trl/blob/main/examples/scripts/sentiment_tuning.py

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="/data/LLM_MODEL/gpt2-imdb", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=128, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=128, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=6, metadata={"help": "kl target for early stopping"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft"})
    use_seq2seq: Optional[bool] = field(default=False, metadata={"help": "whether to use seq2seq models"})
    kl_penalty: Optional[str] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl) and 'mse': mean squared error mse(kl)."
        },
    )
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    output_dir: Optional[str] = field(default="output/ppo_3", metadata={"help": "the output directory"})
    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    kl_penalty=script_args.kl_penalty,
    seed=script_args.seed,
    project_kwargs={'logging_dir':script_args.output_dir}
)

torch.backends.cuda.enable_flash_sdp(True)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = (
    AutoModelForCausalLMWithValueHead if not script_args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=6, input_max_text_length=18):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 500, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
print('Load ref model')
ref_model = trl_model_class.from_pretrained(config.model_name)
device_map = None
peft_config = None


print('Load trainable model')
model = trl_model_class.from_pretrained(
    config.model_name,
    device_map=device_map,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
print(config.model_name)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)


# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
# sentiment_pipe = pipeline("sentiment-analysis", model="output/rm_2/checkpoint-5500", device=device)
print('Reward model')
print(sentiment_pipe.model)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

for epoch, batch in enumerate(ppo_trainer.dataloader):

    '''
    A batch is like 
    {'label': [tensor(1, device='cuda:0')], 
    'input_ids': [tensor([  40, 3505, 4964], device='cuda:0')], 
    'query': ['I remember watching']
    }
    '''
    print(epoch)
    query_tensors = batch["input_ids"]
    
    # print(batch["query"])
    # Get response from gpt2
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # [{'label': 'LABEL_0', 'score': -1.7437267303466797}, {'label': 'LABEL_1', 'score': -2.568110942840576}]

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    assert len(pipe_outputs)==script_args.batch_size
    #print(pipe_outputs[0])
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    #print(rewards)
    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    if epoch %10==0:
        print(epoch, len(ppo_trainer.dataloader))
        print('Q:', batch["query"][0])
        print('R:', batch["response"][0])
        print('Rewards', rewards[0])
    
    if epoch > 10:
        ppo_trainer.log_stats(stats, batch, rewards)
