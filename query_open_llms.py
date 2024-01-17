from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os 
from argparse import ArgumentParser
import torch 
import transformers
import numpy as np 
import pandas as pd
from tqdm import tqdm
from prompts import *



TOKEN = '<HF_TOKEN>'

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()} </s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt



def query_open_llms(args):
    if args.model.startswith('llama2'):
        if args.model=='llama2_7b':
            model = 'meta-llama/Llama-2-7b-chat-hf'
        elif args.model=='llama2_13b':
            model = "meta-llama/Llama-2-13b-chat-hf"
        elif args.model=='llama2_70b':
            model = "meta-llama/Llama-2-70b-chat-hf"
        task = 'text-generation'
    elif args.model.startswith('flan'):
        if args.model=='flan_xxl':
            model = 'google/flan-t5-xxl'
        task = 'text2text-generation'
    elif args.model.startswith('Yi'):
        if args.model=='Yi_34B':
            model = '01-ai/Yi-34B'
        task = 'text-generation'
            
        
    pipeline = transformers.pipeline(
                        task,
                        model=model,
                        torch_dtype=torch.float16,
                        token = TOKEN,
                        device_map="auto",
#                        trust_remote_code=True
                    )
        
    
    for i in np.arange(0,int(args.repeatN)):
        print(f"Running {i}-th trial!")
        result = generate_prompt(args.prompt_task, args.model,args.prompt_type)
        
        for prompt in tqdm(result):
            if args.model.startswith('llama2'):
                messages = [
                    {'role':'user','content':prompt['Question']}]

                query = build_llama2_prompt(messages)
            else:
                query = prompt['Question']

            sample_sequences = pipeline(
            query,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            #eos_token_id = tokenizer.eos_token_id,
            max_length =300,)
            
            
            if args.model.startswith('llama2'):
                prompt['Answer']=sample_sequences[0]['generated_text'][len(query):]
            elif args.model.startswith('flan'):
                prompt['Answer']=sample_sequences[0]['generated_text'][:]
                
            
        df= pd.DataFrame.from_dict(result,orient='columns')
        outdir = f"{args.out_dir}/{args.prompt_task}_{args.model}_{args.prompt_type}_{i}.csv"
        df.to_csv(outdir)
        print(f'Saved to {outdir}!')
        

if __name__ =='__main__':
    parser:ArgumentParser = argparse.ArgumentParser() 

    parser.add_argument('--model', default='llama2_7b')
    parser.add_argument('--out_dir',default='result',help='Save Path')
    parser.add_argument('--prompt_task',default='Anes', help='Choose from:[Anes, MFQ]')
    parser.add_argument('--prompt_type',default='base',help='Choose from:[base,CoT,wInst,wo]')
    parser.add_argument('--prompt_ablation',action='store_true')
    parser.add_argument('--repeatN', default=10, help='Repeat times')
    args = parser.parse_args() 
    
    
    query_open_llms(args)