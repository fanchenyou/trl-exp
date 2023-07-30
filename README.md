## Run RLHF on our tencent cloud server
The repo is rebased from https://github.com/lvwerra/trl/blob/main/examples/scripts/sentiment_tuning

But I pre-downloaded all model and datasets to make it ready to run.

Also add some comments for easier understanding.


### Step-1
``python step_1_sft.py``

Check output at output/sft_1, will be used in step-3.

### Step-2
``python step_2_reward.py``

Check output at output/rm_2, will be used in step-3.


### Step-3
``python step_3_ppo.py``

Use model saved in Step-1 and Step-2 for RL fine-tuning.