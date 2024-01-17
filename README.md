## **Examining Representative Heuristics of LLMs**
The scripts for paper **Examining Representative Heuristics of LLMs**. 

1. `query.py` : code for querying LLMs via API (e.g. openai, gemini)
    - `query_open_llms`: query code for open LLMs e.g. Llama2-70b 
    - `query_misinfo.py`: query code for misinformation detection

2. `extract_scales.py`: extracts scales from the results from `query.py`

3. `result_analysis.ipynb`:
    - generates Fig 2.& 3. in the paper 
    - kernel-of-truth analysis 
    - representative heuristics analysis 

4. `utils.py`: stores functions for preprocessing data, and analysis

#### **Data**
Store the following data under the `Data` directory. 
- Anes: Please refer to https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/ 
- MFQ_Survey_Data
- liar_dataset
