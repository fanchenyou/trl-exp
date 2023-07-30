from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoConfig, AutoModel, AutoTokenizer, Seq2SeqTrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

# read the task at 
# https://huggingface.co/lvwerra/gpt2-imdb-pos
# the purpose of step 1~3 is to generate positive reviews of movies


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="/data/LLM_MODEL/opt-350m", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="/data/LLM_MODEL/imdb", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default='tensorboard', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=384, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output/sft_1", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})


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
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = torch.half

# load model from /data/LLM_MODEL/opt-350m
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=script_args.model_name,
)
print(tokenizer)

# use flash attention
torch.backends.cuda.enable_flash_sdp(True)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)

print(model)

# Step 2: Load the dataset
# the dataset is from https://huggingface.co/datasets/imdb
# pre-downloaded at "/data/LLM_MODEL/imdb"
dataset = load_dataset(path=script_args.dataset_name, split="train")
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 25000
# })

max_seq_length = 384 #script_args.seq_length
dataset_text_field = "text" 

# visualize the data
if 1==2:
    print(dataset[0:10])
    ds0 = dataset[0]
    print(ds0,ds0['text'],len(ds0['text']))

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    save_steps = 1000, # save every 500 iters
)

# Step 4: Define the Trainer
trainer = SFTTrainer(
    model=model,
    #tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    max_seq_length = max_seq_length
)

# Dataset({
#     features: ['input_ids', 'attention_mask'],
#     num_rows: 10852
# }) 10852
print(trainer.train_dataset, len(trainer.train_dataset))

trainer.train()