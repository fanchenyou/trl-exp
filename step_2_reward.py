import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from trl.trainer.reward_trainer import RewardTrainer



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="/data/LLM_MODEL/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="tensorboard", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=384, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output/rm_2", metadata={"help": "the output directory"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # This means: fit the entire model on the GPU:0
    device_map = {"": 0}
else:
    device_map = None
    quantization_config = None
    torch_dtype = torch.half


model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, # "/data/LLM_MODEL/opt-350m",
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)
# print(model)
# exit()

max_seq_length = script_args.seq_length

# Step 2: Load the dataset and pre-process it
# the original dataset is at https://huggingface.co/datasets/Anthropic/hh-rlhf
# I pre-downloaded at /data/LLM_MODEL/hh-rlhf
# each data is a pair of chosen and reject answer
# for training a rewarding model
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, max_seq_length=max_seq_length)
dataset = load_dataset(path="/data/LLM_MODEL/hh-rlhf", split="train")

# use flash attention
torch.backends.cuda.enable_flash_sdp(True)



def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen,max_length=max_seq_length, padding='max_length',
        truncation=True)
        tokenized_k = tokenizer(rejected, max_length=max_seq_length, padding='max_length', 
        truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.max_length
original_columns = dataset.column_names
train_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=16,
)
print(train_dataset)
# train_dataset = train_dataset.filter(
#     lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
#     and len(x["input_ids_rejected"]) <= script_args.seq_length
# )
# print(train_dataset)


# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    save_steps = 2000, # save every 500 iters
)


# Step 4: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    max_length=script_args.seq_length,
)

trainer.train()
