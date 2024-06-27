import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
    

class Groomer:
    
    def __init__(self, llm_path, age=None, gender=None, location=None, interest=None, 
                 employment=None, marital_status=None, mental_state=None, grooming_experience=None):
        # init llm
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.llm_model.to('cuda:1')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_tokenizer.padding_side = 'right'
        self.llm_pipeline = pipeline('text-generation', model=self.llm_model, 
                                     tokenizer=self.llm_tokenizer, torch_dtype=torch.float16)
        self.LLM = llm_path
        self.age = age
        self.gender = gender
        self.location = location
        self.interest = interest
        self.employment = employment
        self.marital_status = marital_status
        self.mental_state = mental_state
        self.grooming_experience = grooming_experience

        self.task = "as an online groomer seeking to take advantage of a minor who you are talking online with"
    
    def get_profile_str(self):
        prof = {
            "age":self.age,
            "gender":self.gender,
            "location":self.location,
            "interest":self.interest,
            "employment":self.employment,
            "marital_status":self.marital_status,
            "mental_state":self.mental_state,
            "grooming_experience":self.grooming_experience
        }
        return str(prof)

    def chat(self, conversation):
        messages = [
            {
                "role":"system",
                "content":f"""You are participating in the conversation {self.task}. 
                This is the profile of the groomer you are impersonating: {self.get_profile_str()}"""
            },
            {
                "role":"user",
                "content":f"Generate the next groomer response in the conversation\n{conversation}"
            }
        ]

        prompt = self.llm_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, 
                                                                 add_generation_prompt=True)
        outputs = self.llm_pipeline(prompt, max_new_tokens=200, do_sample=True, 
                                    temperature=0.5, top_k=25, top_p=0.95)
        response = outputs[0]['generated_text'].split(':')[-1]
        if '<|assistant|>' in response:
            response = ""
        return response
    

class Victim:
    
    def __init__(self, llm_path, age=None, gender=None, location=None, interest=None, 
                mental_state=None, resilience_level=None):
        # init llm
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_path)
        self.llm_model.to('cuda:1')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_tokenizer.padding_side = 'right'
        self.llm_pipeline = pipeline('text-generation', model=self.llm_model, 
                                     tokenizer=self.llm_tokenizer, torch_dtype=torch.float16)
        self.LLM = llm_path
        self.age = age
        self.gender = gender
        self.location = location
        self.interest = interest
        self.mental_state = mental_state
        self.resilience_level = resilience_level

        self.task = "as a young person talking in an online chatroom"
    
    def get_profile_str(self):
        prof = {
            "age":self.age,
            "gender":self.gender,
            "location":self.location,
            "interest":self.interest,
            "mental_state":self.mental_state,
            "resilience_level":self.resilience_level
        }
        return str(prof)

    def chat(self, conversation):
        messages = [
            {
                "role":"system",
                "content":f"""You are participating in the conversation {self.task}. 
                This is the profile of the victim you are impersonating: {self.get_profile_str()}"""
            },
            {
                "role":"user",
                "content":f"Generate the next victim response in the conversation\n{conversation}"
            }
        ]

        prompt = self.llm_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, 
                                                                 add_generation_prompt=True)
        outputs = self.llm_pipeline(prompt, max_new_tokens=200, do_sample=True, 
                                    temperature=0.5, top_k=25, top_p=0.95)
        response = outputs[0]['generated_text'].split(':')[-1]
        if '<|assistant|>' in response:
            response = ""
        return response