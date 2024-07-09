import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
from tqdm import tqdm
from utils.profile import Groomer, Victim, StateAndMentorModel

# load the groomer
groomer_path = "./models/training/groomer_output/checkpoint-6"
grmr = Groomer(groomer_path, 48, "male", "Blacksburg, Virginia", "cars", 
               "Nurse", "un-married", "Loneliness", "Medium")
# load the victim
victim_path = "./models/training/victim_output/checkpoint-6"
vctm = Victim(victim_path, 13, "female", "Roanoke, Virginia", "dresses",
              "Inferiority", "Low")

stmo = StateAndMentorModel('mistralai/Mistral-7B-Instruct-v0.2')

NUM_CONVERSATION_TURNS = 120
NUM_CONVERSATIONS = 30
MENTOR_WINDOW = 7
mentor_buffer = []

for num_conv in tqdm(range(1, NUM_CONVERSATIONS+1)):
    conv = ["START","groomer:hi how old are you? male or female?"]
    fil = open(f'./out/synth_conv_{num_conv}_nostate.txt', 'w')
    fil.write(f'{conv[1]}\n')
    mentor_buffer.append(f"groomer:{conv[1]}")

    adv_fil = open(f'./adv_out/conv_{num_conv}_nostate.txt', 'w')

    for i in tqdm(range(NUM_CONVERSATION_TURNS+1)):
        print(conv)
        # victim takes turn
        if len(conv) > 12:
            conversation = ';;'.join(conv[-12:])
        else:
            conversation = ';;'.join(conv)


        v_response = vctm.chat(conversation)
        conv.append(f"victim:{v_response}")
        mentor_buffer.append(f"victim:{v_response}")
        # conv.append(f"{role}:{response}")

        if len(conv) == NUM_CONVERSATION_TURNS+1:
            break


        # groomer takes turn
        if len(conv) > 12:
            conversation = ';;'.join(conv[-12:])
        else:
            conversation = ';;'.join(conv)

        #g_response = grmr.chat(conversation, stmo.get_current_goal())
        g_response = grmr.chat_nogoal(conversation)
        conv.append(f"groomer:{g_response}")
        mentor_buffer.append(f"groomer:{g_response}")
        # conv.append(f"{role}:{response}")

        if len(conv) == NUM_CONVERSATION_TURNS+1:
            break
        if len(conv) > 7:
            cn_state = stmo.determine_state_change(conv[-7:])
        else:
            cn_state = stmo.determine_state_change(conv)

        if len(mentor_buffer) > MENTOR_WINDOW:
            # mentor the victim
            advice = stmo.mentor(mentor_buffer)

            # write the convo and advice to a file
            adv_fil.write(f'conversation: {str(mentor_buffer)}\n')
            adv_fil.write(f'advice: {advice}\n')
            adv_fil.write('\n')

            # empty buffer
            mentor_buffer = []
        
        fil.write(f'victim:{v_response}\ngroomer:{g_response}\nstate: {str(cn_state)}\n')
    mentor_buffer = []
    stmo.reset()
    fil.close()
    adv_fil.close()
    
    # save the conversation history
    # open(f'./out/synth_conv_{num_conv}.txt', 'w').writelines([c+'\n' for c in conv])