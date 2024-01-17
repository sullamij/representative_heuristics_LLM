from argparse import ArgumentParser
import argparse
import os 
import pandas as pd 
import numpy as np 
import re 
import ast

def extract_scales(df,patterns): 
    Scale=[]
    for i, row in df.iterrows():
        #print(row['Answer'])
        scale=''
        for pattern in patterns: 
            res = re.findall(pattern, row['Answer'])
            if res: 
                if type(res[0])==tuple:
                    if pattern in [r'Scale:\s+(\d)-(\d)',r'Scale:\s+(\d)\sor\s(\d)']:
                        scale = (int(res[0][0])+int(res[0][1]))/2
                    else:
                        scale = res[0][0]
        #            print(res[0][0])
                else: 
                    scale =res[0]
        #            print(res[0])
                #print(pattern)
                Scale.append(scale)
                break

        if scale=='':
            Scale.append(scale)

    df['Scale'] = Scale 
    return df 




if __name__=='__main__':
    parser:ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--outidr',default='result/')
    parser.add_argument('--task',default='Anes',help='Choose from: [Anes,MFQ]')
    parser.add_argument('--model',default='gpt-3.5-turbo')
    args = parser.parse_args()

    outdir = args.outidr
    task = args.task 
    model = args.model

    filelists = os.listdir(outdir)
    files = [f for f in filelists if f.startswith(task)]
    files = [f for f in files if model in f]
    patterns = [r'Scale:\s+.*\s+(\d)',r'Scale:\s+(\d)-(\d)',r'Scale:\s+(\d)\sor\s(\d)',r'Scale:\s+(\b\d+([\.,]\d+)?)',r'Scale:(\b\d+([\.,]\d+)?)'] # patterns for gpt-3.5, gpt-4, llama2-70b


    for file in files:
        print(f'Extracting {file}') 
        df=pd.read_csv('result/'+file)
        df=extract_scales(df, patterns)
        df.to_csv('result/'+file)
        print(f"Extracted result/{file}")



