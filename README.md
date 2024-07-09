# RCSAC: Improving Adolescents’ Risk-Coping Skills Against Cybergrooming Using Conversational Agents

## Important Files
|Filename|Purpose|
|-----|-----|
|simulate_conversation.py|Generates a text file conversation between the two LLMs.|
|utils/profile.py|Object oriented classes to interface with each of the LLMs.|
|data/clean_data.ipynb| The beginning work that has been done to clean some of the PJ dataset. The initial cleaning was inadequate for training the models, so the cleaning was revisited within this file.|
|data/create_train_files.ipynb|Creates the files used for training.|
|data/groomer_[test/train].csv|The training and testing data for the groomer.|
|data/victim_[test/train].csv|The training and testing data for the victim.|
|models/training/finetune.py|This is the primary finetuning file that you should use. It has been equipped to train either the groomer or victim based upon the arguments within the command line.|

## Usage
### Finetuning
Finetuning a model can be achieved by running the command

```bash
python finetune.py --role [groomer/victim]
```

I would recommend using the `CUDA_VISIBLE_DEVICES` command before the 'python' in order to signal to the program which device should be used.

### Simulating a Conversation
To simulate many conversations between both the victim and groomer LLMs, use the command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python simulate_conversation.py
```

To simulate a conversation, it is required that at least 3 GPUs are used. Each model takes ~28GB each.

To modify the number of turns that each conversation is, and how many conversations are synthesizes, modify lines 21-22 in `simulate_conversation.py`:

```python
stmo = StateAndMentorModel('mistralai/Mistral-7B-Instruct-v0.2')

NUM_CONVERSATION_TURNS = 120
NUM_CONVERSATIONS = 30
MENTOR_WINDOW = 7
mentor_buffer = []

for num_conv in tqdm(range(1, NUM_CONVERSATIONS+1)):
    ...
```

#### Modifying Model Parameters
All of the model settings, parameters, etc. exist within the `data/profile.py` file. It is here that you can manipulate the starting category, how many turns the mentor looks at to determine the feedback, etc. These profiles are not used in the finetuning, but are the primary method of interaction for the models in the conversation simulation and in the [RCSAC-Interface][https://github.com/TrevorAshby/RCSAC-Interface]. 

The current implementation begins the groomer at category A4, as this increases the rate at which the groomer will attempt to groom. Categories A1-A4 take a significant time to develop. This will be addressed in future research.