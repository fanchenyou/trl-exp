import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, TrainingArguments, HfArgumentParser
from datasets import load_dataset, Dataset
from datasets import DatasetDict
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from trl.trainer.ppo_trainer import PPOConfig, PPOTrainer
from trl.trainer.sft_trainer import SFTTrainer
from trl.core import LengthSampler
from dataclasses import dataclass, field
from typing import Optional
import numpy as np



# step-4 performs reject sampling
# only use most positive reviews for SFT


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # model_name: Optional[str] = field(
    #     default="output/sft_1/checkpoint-3500", metadata={"help": "the model name"})
    model_name: Optional[str] = field(
        default="output/rs_4/checkpoint-4", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(
         default="output/rm_2/checkpoint-16000", metadata={"help": "the model name"})
    # reward_model_name: Optional[str] = field(
    #     default="lvwerra/distilbert-imdb", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default='tensorboard', metadata={
                                    "help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(
        default=128, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(
        default=128, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(
        default=6, metadata={"help": "kl target for early stopping"})
    use_peft: Optional[bool] = field(default=False, metadata={
                                     "help": "whether to use peft"})
    use_seq2seq: Optional[bool] = field(
        default=False, metadata={"help": "whether to use seq2seq models"})
    seed: Optional[int] = field(
        default=0, metadata={"help": "the random seed"})
    output_dir: Optional[str] = field(
        default="output/rs_4", metadata={"help": "the output directory"})



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.enable_flash_sdp(True)
torch.set_float32_matmul_precision("medium")

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True,
               "function_to_apply": "none", "batch_size": 16}

trl_model_class = (
    AutoModelForCausalLMWithValueHead if not script_args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
)

trl_model_class = AutoModelForCausalLMWithValueHead
batch_size = 128



# load in model and reference model
print('Load trainable model')
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name).to(device)
print('Load reward model')
reward_pipe = pipeline("sentiment-analysis", model=script_args.reward_model_name, device=device)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Build data like in step_3
# since we want to have review text
def build_dataset(dataset_name="/data/LLM_MODEL/imdb", input_min_text_length=6, input_max_text_length=18):
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
    # tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(
            sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset()

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# We use ppo_trainer to generate query pairs.
# But we implement reject sampling instead of PPO RL.

learning_rate = 1.41e-5


config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=1.41e-5,
    log_with='tensorboard',
    mini_batch_size=batch_size,
    batch_size=batch_size,
    early_stopping=False,
    kl_penalty="kl",
    seed=123,
    project_kwargs={'logging_dir': script_args.output_dir}
)

ppo_trainer = PPOTrainer(config, ref_model, None,
                         tokenizer, dataset=dataset, data_collator=collator)


# Define the SFT training arguments, as in step_1
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=learning_rate,
    report_to='tensorboard',
    logging_steps=1,
    save_steps = 1, # save every 500 iters
    save_total_limit = 3, # only save most recent 3 checkpoints, to avoid exceeding disk
    num_train_epochs = 1,
    max_steps=-1,
)

# features = {}
# features['text'] = ['I'*25]
# ds = Dataset.from_dict(features)
# ds.set_format(type="torch")

sft_trainer = SFTTrainer(
        model=script_args.model_name,
        #tokenizer=tokenizer,
        args=training_args,
        max_seq_length=384,
        train_dataset=None,
        dataset_text_field="text"
    )


model,optimizer,tokenizer = sft_trainer.model,sft_trainer.optimizer,sft_trainer.tokenizer


gen_kwargs = {
    "min_length": -1, 
    "top_k": 0.0, 
    "top_p": 1.0, 
    "do_sample": True, 
    "pad_token_id": tokenizer.eos_token_id
}

N_BEST_OF = 4


# dataset for mini-batch 
# train SFT
features = {'text':[],'score':[]}

T = 0

while T<10:
    for epoch, batch in enumerate(ppo_trainer.dataloader):

        '''
        A batch is like 
        {'label': [tensor(1, device='cuda:0')], 
        'input_ids': [tensor([  40, 3505, 4964], device='cuda:0')], 
        'query': ['I remember watching']
        }
        '''
        print(epoch)
        # a list of input_ids
        query_tensors = batch["input_ids"]

        output_data = dict()
        output_data["query"] = batch["query"]
        query_tensors = batch["input_ids"]

        # keep track of the generated answers
        response_tensors = []
        response_tensors_ref = []
        response_tensors_best_of = []

        train_time = 0
        output_length_sampler = LengthSampler(4, 16)
        
        for i in range(len(query_tensors)):
            gen_len = output_length_sampler()

            query = query_tensors[i]
            query_word = tokenizer.decode(query)
            #print(query)

            if epoch % 2 ==0:
                # generate from model
                output = model.generate(query.unsqueeze(dim=0), max_new_tokens=gen_len, **gen_kwargs).squeeze()
                output_word = tokenizer.decode(output)
                #print(query_word, '----', output_word)
                response_tensors.append(output_word)

            # generating copies of the same query for the Best-of-n sampling
            queries = query.repeat((N_BEST_OF, 1))
            output_ref = ref_model.generate(queries.to(device), max_new_tokens=gen_len, **gen_kwargs).squeeze()
            # print(output_ref)
            output_ref_word = tokenizer.batch_decode(output_ref)
            # print(query_word)
            # print(output_ref_word)
            # print()
            response_tensors_best_of.append(output_ref_word)
            # just choose one as a single output of the reference model
            response_tensors_ref.append(output_ref_word[0])

            # if i>30:
            #     break

        if epoch % 2 ==0:
            scores_ref = [output[0]["score"] for output in reward_pipe(response_tensors_ref, **sent_kwargs)]
            scores = [output[0]["score"] for output in reward_pipe(response_tensors, **sent_kwargs)]
            logs = {'score_model': np.mean(scores), 'score_ref': np.mean(scores_ref)}
            ppo_trainer.accelerator.log(logs, step=epoch)

            if epoch % 4 ==0:
                print('==============================================')
                print('----------------  Trained model --------------', np.average(np.mean(scores)))
                for sc, st in zip(scores[:5], response_tensors[:5]):
                    print(sc, st)
                print()
                print('----------------  Ref model --------------', np.mean(scores_ref))
                for sc, st in zip(scores_ref[:5], response_tensors_ref[:5]):
                    print(sc, st)
                print()

        # scores_best_of = []
        for i, response in enumerate(response_tensors_best_of):
            # base_score = scores_ref[i]
            scores = torch.tensor([output[0]["score"] for output in reward_pipe(response,  **sent_kwargs)])
            #print(scores,'---',response)
            t = torch.argmax(scores)
            if scores[t]>0:
                #print(response[t],'====')
                features['text'].append(response[t])
                #features['labels'].append(0)
                features['score'].append(scores[t].item())

        if len(features['text'])>=32:
            avg_score = np.mean(features['score'])
            del features['score']
            training_pos_dataset = Dataset.from_dict(features)
            training_pos_dataset.set_format(type="torch")


            print('[Training] %d, dataset size %d, avg score %.3f' % (train_time, len(training_pos_dataset), avg_score))
            sft_trainer_tmp = SFTTrainer(
                    model=model,
                    optimizers=(optimizer,None),
                    tokenizer=tokenizer,
                    args=training_args,
                    max_seq_length=384,
                    train_dataset=training_pos_dataset,
                    dataset_text_field="text",
                )
            

            ##### perform training #####
            sft_trainer_tmp.train()

            train_time+=1
            features = {'text':[],'score':[]}
            torch.cuda.empty_cache()

            model = sft_trainer_tmp.model
            optimizer = sft_trainer_tmp.optimizer


    