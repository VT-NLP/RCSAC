import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
from tqdm import tqdm
from utils.profile import Groomer, Victim

# load the groomer
groomer_path = "./models/training/groomer_output/checkpoint-370"

# groomer_model = AutoModelForCausalLM.from_pretrained(groomer_path)
# groomer_tokenizer = AutoTokenizer.from_pretrained(groomer_path)
grmr = Groomer(groomer_path, 48, "male", "Blacksburg, Virginia", "cars", 
               "Nurse", "un-married", "Loneliness", "medium")
# groomer_model.to('cuda:1')

# groomer_tokenizer.pad_token = groomer_tokenizer.eos_token
# groomer_tokenizer.padding_side = 'right'

# pipe_groomer = pipeline("text-generation", model=groomer_model, tokenizer=groomer_tokenizer, torch_dtype=torch.float16)

# load the victim
victim_path = "./models/training/victim_output/checkpoint-550"

# victim_model = AutoModelForCausalLM.from_pretrained(victim_path)
# victim_tokenizer = AutoTokenizer.from_pretrained(victim_path)
vctm = Victim(victim_path, 13, "female", "Roanoke, Virginia", "dresses",
              "Inferiority")
# victim_model.to('cuda:1')

# victim_tokenizer.pad_token = victim_tokenizer.eos_token
# victim_tokenizer.padding_side = 'right'

# pipe_victim = pipeline("text-generation", model=victim_model, tokenizer=victim_tokenizer, torch_dtype=torch.float16)


# roles = ["victim", "groomer"]
# role_id = 1


NUM_CONVERSATION_TURNS = 100
NUM_CONVERSATIONS = 5

for num_conv in tqdm(range(1, NUM_CONVERSATIONS+1)):
    conv = ["START","groomer:hi how old are you? male or female?"]
    for i in tqdm(range(NUM_CONVERSATION_TURNS+1)):
        print(conv)
        # victim takes turn
        if len(conv) > 9:
            conversation = ';;'.join(conv[-9:])
        else:
            conversation = ';;'.join(conv)

        # role_id = not role_id
        # role = roles[role_id]
        # task = "as a young person talking in an online chatroom"

        # messages = [
        #     {
        #     "role":"system",
        #     "content": f"You are participating in the conversation {task}. "
        #     },
        #     {
        #     "role": "user",
        #     "content": f"Generate the next {role} response in the conversation:\n{conversation}"
        #     }
        # ]
        # add to conv
        # prompt = pipe_victim.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = pipe_victim(prompt, max_new_tokens=200, do_sample=True, temperature=0.5, top_k=25, top_p=0.95)
        # response = outputs[0]['generated_text'].split(':')[-1]
        # if '<|assistant|>' in response:
        #     response = ""
        response = vctm.chat(conversation)
        conv.append(f"victim:{response}")
        # conv.append(f"{role}:{response}")

        if len(conv) == NUM_CONVERSATION_TURNS+1:
            break


        # groomer takes turn
        if len(conv) > 9:
            conversation = ';;'.join(conv[-9:])
        else:
            conversation = ';;'.join(conv)

        # role_id = not role_id
        # role = roles[role_id]
        # task = "as an online groomer seeking to take advantage of a minor who you are talking online with"

        # messages = [
        #     {
        #     "role":"system",
        #     "content": f"You are participating in the conversation {task}. "
        #     },
        #     {
        #     "role": "user",
        #     "content": f"Generate the next {role} response in the conversation:\n{conversation}"
        #     }
        # ]
        # add to conv
        # prompt = pipe_groomer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = pipe_groomer(prompt, max_new_tokens=200, do_sample=True, temperature=0.5, top_k=25, top_p=0.95)
        # response = outputs[0]['generated_text'].split(':')[-1]
        # if '<|assistant|>' in response:
        #     response = ""
        response = grmr.chat(conversation)
        conv.append(f"groomer:{response}")
        # conv.append(f"{role}:{response}")

        if len(conv) == NUM_CONVERSATION_TURNS+1:
            break

    # save the conversation history
    open(f'./out/synth_conv_{num_conv}.txt', 'w').writelines([c+'\n' for c in conv])