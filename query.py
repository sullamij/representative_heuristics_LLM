import argparse
import os 
from argparse import ArgumentParser
from prompts import generate_prompt
import random 
import time 
import json
import pandas as pd  
import requests

from tqdm import tqdm
import pickle 
import pandas as pd
import openai
import backoff
import numpy as np 
from datetime import datetime 
import google.generativeai as genai 


OPENAI_API_KEY = '<KEY>'
GOOGLE_GEN_API_KEY='<GOOGLE_KEY>'


# python -m query --api openai --model gpt-3.5-turbo --prompt_task Anes
# python -m query --api gemini --model bard --prompt_task Anes



def query_gemini(args):
    genai.configure(api_key=GOOGLE_GEN_API_KEY)
    print(f"Running {args.model}")
    prompts=generate_prompt(args.prompt_task, args.model, args.prompt_type)
    result = [] 
    model = genai.GenerativeModel('gemini-pro')
    for i, prompt in tqdm(enumerate(prompts)):
        try:
            response = model.generate_content(prompt['Question'])
            prompt['Answer']=response.text
            result.append(prompt)
        except:
            print(f"Exception occured! {i}-th")
            if response.prompt_feedback.block_reason:
                prompt['Answer']='NaN'
                result.append(prompt)
            else:
                print('Error Occured')
                prompt['Answer']='Error'
                result.append(prompt)
                time.sleep(60)
    now = datetime.now()
    timestamp = now.strftime("%m:%d:%Y:%H:%M:%S")
    out_dir = f"{args.out_dir}/{args.prompt_task}_{args.model}_{args.prompt_type}_{timestamp}.csv"
    df = pd.DataFrame.from_dict(result,orient='columns')
    df.to_csv(out_dir)
    print("++++Saved!++++")

def query_openai(args):
    openai.api_key = OPENAI_API_KEY
    print(f":::RUNNING {args.model}::::")

    promptss = generate_prompt(args.prompt_task, args.model,args.prompt_type)
    result = [] 

    for i, prompt in tqdm(enumerate(promptss)):
        messages = [{'role':'user','content':prompt['Question']}]
        response= query(args.model,messages)
        prompt['Answer']=response['choices'][0]['message']['content']
        result.append(prompt)
        
    now = datetime.now() 
    timestamp = now.strftime("%m:%d:%Y:%H:%M:%S")
    out_dir = f"{args.out_dir}/{args.prompt_task}_{args.model}_{args.prompt_type}_{timestamp}.csv"
    df = pd.DataFrame.from_dict(result,orient='columns')
    df.to_csv(out_dir)
    print("++++Saved!+++++")

@backoff.on_exception(backoff.expo,openai.error.APIError)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def query(model,messages):
    response = openai.ChatCompletion.create(
        model = model, 
        messages = messages
    )
    return response
    


if __name__ =='__main__':
    parser:ArgumentParser = argparse.ArgumentParser() 

    parser.add_argument('--api', default='openai', help='Select from: [bard,openai]')
    parser.add_argument('--model', default='gpt-3.5-turbo')
    parser.add_argument('--interim_file',help='directory to store interim file')
    parser.add_argument('--out_dir',default='result',help='Save Path')
    parser.add_argument('--prompt_task',default='Anes', help='Choose from:[Anes, MFQ]')
    parser.add_argument('--prompt_type',default='base',help='Choose from:[base,CoT,wInst,wo]')
    parser.add_argument('--prompt_ablation',action='store_true')
    parser.add_argument('--repeatN', default=10, help='Repeat times')
    args = parser.parse_args() 


    if args.api=='openai':
#        if args.prompt_ablation: 
            ### ablation ### 
#        else:
        queryfn=query_openai
    
    elif args.api=='gemini':
        queryfn=query_gemini

    for i in np.arange(0,int(args.repeatN)):
        queryfn(args)



