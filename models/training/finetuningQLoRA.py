# helpful resource from: https://medium.com/data-science-in-your-pocket/lora-for-fine-tuning-llms-explained-with-codes-and-example-62a7ac5a3578

# finetune vicuna using QLoRa
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
data_files = {'train':'groomer_train.json', 'test':'groomer_test.json'}
dataset = load_dataset('json', data_files=data_files)

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to('cuda:0')

# Makes training faster but a little less accurate
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)

# setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

def prompt_instruction_format(sample, groomer=True):
  task = ""
  role = ""
  if groomer:
    task = "as an online groomer seeking to take advantage of a minor who you are talking online with"
    role = "groomer"
  
  messages = [
    {
      "role":"system",
      "content": f"You are participating in the conversation {task}. "
    },
    {
      "role": "user",
      "content": f"Generate the next response in the conversation: {sample['dialogue']}\n{role}:{sample['response']}"
    }
  ]

  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
  return prompt

#   return f"""### Instruction:
#     Use the Task below and the Input given to write the Response:

#     ### Task:
#     Generate the next response to the dialogue

#     ### Input:
#     {sample['dialogue']}

#     ### Response:
#     {sample['response']}
#     """ 

# Create the trainer
trainingArgs = TrainingArguments(
    output_dir='output',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    learning_rate=2e-4
)

peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['groomer_train'],
    eval_dataset = dataset['groomer_test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format,
    args=trainingArgs,
)

trainer.train()

