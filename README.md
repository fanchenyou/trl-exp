## Run RLHF on our tencent cloud server
The repo is rebased from https://github.com/lvwerra/trl/blob/main/examples/scripts/sentiment_tuning

But I pre-downloaded all model and datasets on our server to make it ready to run.

Also add some comments for easier understanding.

### Questions to think
1. What is the model of a facebook/opt-350m? THUDM/ChatGLM? 
Print them and show. 
2. What is the building block of an LLM? What is GLU block? What is attention? 
3. What are the SFT_Trainer, Reward_Trainer, PPO_Trainer working in details? Find out the loss functions. 
4. Visualize the training process with ***tensorboard***. In VSCode, open the command palette (Ctrl/Cmd + Shift + P) , search for the command “Python: Launch TensorBoard” and press enter. 

### Step-0 Task Understanding: generating positive movie reviews
Read the task at https://huggingface.co/lvwerra/gpt2-imdb-pos

### Step-1 Supervised fine-tuning on IMDB movie data
``python step_1_sft.py``

Check output at output/sft_1, will be used in step-3.



### Step-2 Reward modeling as sentimental classification of a movie review
``python step_2_reward.py``

Check output at output/rm_2, will be used in step-3.


### Step-3 RL to encourages a model to generate only positive review 
``python step_3_ppo.py``

Use model saved in Step-1 and Step-2 for RL fine-tuning.

