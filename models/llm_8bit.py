from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# configs following:
#https://github.com/gmongaras/Llama-2_Huggingface_4Bit_QLoRA/blob/main/main.py

def get_llm(name, use_lora, lora_r, lora_alpha, mode):
    llm_tokenizer = AutoTokenizer.from_pretrained(name)
    if mode == 'train':
        print('llm load in train mode')
        llm_model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.bfloat16
            )
        llm_model = prepare_model_for_kbit_training(llm_model)
    else:
        print('llm load in test mode')
        # llm_model2 = AutoModelForCausalLM.from_pretrained(
        llm_model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=False,
            torch_dtype=torch.bfloat16
            )

    if use_lora:
        peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules="all-linear",
                bias="none", 
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
        llm_model = get_peft_model(llm_model, peft_config)
        llm_model.print_trainable_parameters()
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # for llama3.1-8b
    return llm_tokenizer, llm_model