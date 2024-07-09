import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
import argparse


def prompt_instruction_format_groomer(sample):
    task = "as an online groomer seeking to take advantage of a minor who you are talking online with"
    role = "groomer"

    messages = [
    {
        "role":"user",
        "content": f"You are participating in the conversation {task}. "
    },
    {
        "role":"assistant",
        "content": f"I will respond with only 1 or 2 sentences."
    },
    {
        "role": "user",
        "content": f"Generate the next {role} response in the conversation:\n{sample['dialogue']}"
    },
    {
        "role": "assistant",
        "content": f"{role}:{sample['response']}"
    }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt

def prompt_instruction_format_victim(sample):
    task = "as a young person talking in an online chatroom"
    role = "victim"

    messages = [
    {
        "role":"user",
        "content": f"You are participating in the conversation {task}. "
    },
    {
        "role":"assistant",
        "content": f"I will respond with only 1 or 2 sentences."
    },
    {
        "role": "user",
        "content": f"Generate the next {role} response in the conversation:\n{sample['dialogue']}"
    },
    {
        "role": "assistant",
        "content": f"{role}:{sample['response']}"
    }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt

# --------------- MAIN ---------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set flags
    parser.add_argument("-role", "--role", help="Role to use (groomer, victim)")
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    # setting padding instructions for tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",torch_dtype=torch.float32, device_map='auto')
    # Makes training faster but a little less accurate
    model.config.pretraining_tp = 1

    data_files = {'groomer_train':'../../data/groomer_train_old.csv', 
                'groomer_test':'../../data/groomer_test_old.csv', 
                'victim_train':'../../data/victim_train_old.csv',
                'victim_test':'../../data/victim_test_old.csv'}
    dataset = load_dataset('csv', data_files=data_files, delimiter='|', column_names=['dialogue', 'response'])

    role = args.role

    # Create the trainer
    trainingArgs = TrainingArguments(
        output_dir=f'{role}_output',
        num_train_epochs=6,
        per_device_train_batch_size=1,
        save_strategy="epoch",
        learning_rate=5e-4
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if role == "victim":
        form_func = prompt_instruction_format_victim
    else:
        form_func = prompt_instruction_format_groomer

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset[f'{role}_train'],
        eval_dataset = dataset[f'{role}_test'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=form_func,
        args=trainingArgs,
    )

    trainer.train()