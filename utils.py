import pandas as pd 
import os
import ast
import numpy as np
from statistics import mean 
from sympy import symbols, solve, Eq

base_columns={'VCF0004':'YEAR','VCF0303':'PARTY'}
columns_of_interest = {'VCF0803':'Lib_Con','VCF0503':'Lib_Con_D', 'VCF0504':'Lib_Con_R',
                       'VCF0806':'Gov_HealthInsurance','VCF0508':'Gov_HealthInsurance_D','VCF0509':'Gov_HealthInsurance_R',
                       'VCF0809':'Job_Income','VCF0513':'Job_Income_D','VCF0514':'Job_Income_R',
                       'VCF0811':'Urban_unrest','VCF0528':'Urban_unrest_D','VCF0529':'Urban_unrest_R',
                       'VCF0817':'School_bus','VCF0533':'School_bus_D','VCF0534':'School_bus_R',
                       'VCF0830':'Aid_Black','VCF0517':'Aid_Black_D','VCF0518':'Aid_Black_R',
                       'VCF0832':'Right_Accused','VCF0524':'Right_Accused_D','VCF0525':'Right_Accused_R',
                       'VCF0834':'Women_equal','VCF0537':'Women_equal_D','VCF0538':'Women_equal_R',
                       'VCF0839':'Gov_services','VCF0541':'Gov_services_D','VCF0542':'Gov_services_R',
                       'VCF0843':'Defense_Spending','VCF0549':'Defense_Spending_D','VCF0550':'Defense_Spending_R',
                       'VCF0838':'Abortion','VCF9234':'Abortion_D','VCF9235':'Abortion_R' # 4 scale 
                       }

columns_thermometer = {'VCF9267':'Muslims','VCF9268':'Rich','VCF9269':'Christians',
                       'VCF0201':'Democrats','VCF0202':'Republicans','VCF0203':'Protestants',
                       'VCF0204':'Catholics','VCF0205':'Jews','VCF0206':'Blacks',
                       'VCF0207':'Whites','VCF0208':'Southerners','VCF0209':'BigBusiness',
                       'VCF0210':'LaborUnions','VCF0211':'Liberals','VCF0212':'Conservatives'}

columns_hard_lazy ={'VCF9270':'HL_Whites','VCF9271':'HL_Blacks','VCF9272':'HL_HispanicAmerican','VCF9273':'HL_AsianAmerican'}

def get_emp_anes():
    anes_file = 'Data/anes/anes_timeseries_cdf_csv_20220916.csv'
    df = pd.read_csv(anes_file)
    df_renamed=df.rename(columns=base_columns)
    df_renamed=df_renamed.rename(columns=columns_of_interest)
    return df_renamed

def make_mean_df(df,cols):

    Rows=[]

    for colname in cols:

        df[colname]=df[colname].astype('string')

        ### Pre-process #### 
        df_filtered = df.dropna(subset=[colname])
        mask = df_filtered[colname]==' '
        df_filtered=df_filtered[~mask]
        filtered=df_filtered.groupby(['PARTY',colname]).count().reset_index()[['PARTY',colname,'Version']].query("PARTY==1 or PARTY==3")
        filtered['PARTY']=filtered['PARTY'].replace([1,3],['Democrats','Republicans'])
        #filtered['PARTY']=filtered['PARTY'].astype('string')
        filtered=filtered[~(filtered[colname].isin(['0','9']))]
        if colname=='Gov_services':
            filtered[colname]=filtered[colname].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])
        if colname=='Abortion':
            filtered[colname]=filtered[colname].replace(['1','2','3','4'],['4','3','2','1'])
        filtered['Normalized']=filtered['Version']/filtered.groupby('PARTY')['Version'].transform('sum')
        #filtered[colname].astype(int)
        filtered[colname]=filtered[colname].astype(int)
        demo_std=np.std(np.repeat(filtered.query("PARTY=='Democrats'")[colname].tolist(), filtered.query("PARTY=='Democrats'")['Version'].tolist()))
        repub_std=np.std(np.repeat(filtered.query("PARTY=='Republicans'")[colname].tolist(), filtered.query("PARTY=='Republicans'")['Version'].tolist()))

        filtered['Mean']= filtered[colname]*filtered['Normalized']
        filtered['Avg_Total'] = filtered.groupby('PARTY')['Mean'].transform('sum')

        Rows.append(['Democrats',colname,filtered.query("PARTY=='Democrats'")['Avg_Total'].unique()[0],demo_std])
        Rows.append(['Republicans',colname,filtered.query("PARTY=='Republicans'")['Avg_Total'].unique()[0],repub_std])

    rdf=pd.DataFrame(columns=['Party','Topic','Scale','Std'],data=Rows)
    rdf['Model']=['Empirical']*len(rdf)

    return rdf

def get_anes_responses():
    Results = [] 

    models = ['llama2_70b','gpt-3.5','gpt-4','gemini']
    #model_map={'llama2_70b':'Llama2-70b','gpt-3.5':'Gpt-3.5','gpt-4':'Gpt-4','bard':'Bard'}
    model_map={'llama2_70b':'Llama2-70b','gpt-3.5':'Gpt-3.5','gpt-4':'Gpt-4','gemini':'Gemini'}

    for model in models: 
        print(f'Process...{model}')
        processed = 'result/processed/'
        task = 'Anes'
        filelists=os.listdir(processed)
        files=[f for f in filelists if f.startswith(task)]
        files=[f for f in files if model in f]

        responses = []
        for file in files:
            dff=pd.read_csv(processed+file) 
            responses.append(dff)    
        responses=pd.concat(responses,ignore_index=True)

        if model =='bard':
            mask=responses['Scale']=='[]'
            responses=responses[~mask]
            responses['Scale']=responses['Scale'].apply(lambda x: ast.literal_eval(x))
            responses['Scale']=responses['Scale'].apply(lambda x: [float(t) for t in x])
            responses['Scale'] = responses['Scale'].apply(lambda x: mean(x))
        
        responses=responses.dropna(subset='Scale')
        responses['Scale'] = responses['Scale'].astype(int)

        mean=responses.groupby(['Key']).mean(['Scale']).reset_index()['Scale'].tolist()
        std=responses[['Scale','Key']].groupby('Key').std().reset_index()['Scale'].tolist()
        keys=responses.groupby(['Key']).mean(['Scale']).reset_index()['Key'].tolist()

        #responses=responses.groupby(['Key']).mean(['Scale']).reset_index() 
        party = [] 
        topic = [] 

        for i, row in enumerate(keys):
            if row.endswith('D'):
                party.append("Democrats")
                topic.append(row.split('_D')[0])
            else:
                party.append('Republicans')
                topic.append(row.split('_R')[0])
        
        #responses['Party']=party
        #responses['Topic']=topic
        #responses=responses[['Party','Topic','Scale']]
        models=[model_map[model]]*len(keys)
        #responses['Model']=[model_map[model]]*len(responses)
        dic = {'Party':party,'Topic':topic,'Scale':mean,'Std':std,'Model':models}
        responses=pd.DataFrame(dic)
        Results.append(responses)
    responses=pd.concat(Results,ignore_index=True)
    return responses

def get_empirical_DR(df,cols):
    Rows = [] 

    for colname in cols: 
       df[colname+'_D']=df[colname+'_D'].astype('string')
       df[colname+'_R']=df[colname+'_R'].astype('string')

       df_filter_D = df.dropna(subset=[colname+'_D'])
       mask = df_filter_D[colname+'_D']==' '
       df_filter_D=df_filter_D[~mask]
       df_filter_D = df_filter_D[~(df_filter_D[colname+'_D'].isin(['0','9','8','-8','-9']))]

       df_filter_R = df.dropna(subset=[colname+'_R'])
       mask = df_filter_R[colname+'_R']==' '
       df_filter_R=df_filter_R[~mask]
       df_filter_R = df_filter_R[~(df_filter_R[colname+'_R'].isin(['0','9','8','-8','-9']))]

       if colname=='Gov_services':
            df_filter_D[colname+'_D']=df_filter_D[colname+'_D'].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])
            df_filter_R[colname+'_R']=df_filter_R[colname+'_R'].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])

       if colname=='Abortion':
            df_filter_D[colname+'_D']=df_filter_D[colname+'_D'].replace(['1','2','3','4'],['4','3','2','1'])
            df_filter_R[colname+'_R']=df_filter_R[colname+'_R'].replace(['1','2','3','4'],['4','3','2','1'])

       df_filter_R=df_filter_R.groupby([colname+'_R']).count().reset_index()[[colname+'_R','Version']]
       #df_filter_R['Normalized']=df_filter_R['Version']/
       df_filter_D=df_filter_D.groupby([colname+'_D']).count().reset_index()[[colname+'_D','Version']]

       R_std = np.std(np.repeat(df_filter_R[colname+'_R'].astype(int).tolist(),df_filter_R['Version']))
       D_std = np.std(np.repeat(df_filter_D[colname+'_D'].astype(int).tolist(),df_filter_D['Version']))


       df_filter_R['Normalized'] = df_filter_R['Version']/df_filter_R['Version'].sum()
       df_filter_D['Normalized'] = df_filter_D['Version']/df_filter_D['Version'].sum()

       df_filter_R[colname+'_R'] = df_filter_R[colname+'_R'].astype(int)
       df_filter_D[colname+'_D'] = df_filter_D[colname+'_D'].astype(int)

       df_filter_R['Mean'] = df_filter_R[colname+'_R']*df_filter_R['Normalized']
       df_filter_D['Mean'] = df_filter_D[colname+'_D']*df_filter_D['Normalized']


       Rows.append({'Topic':colname, 'Party':'Republicans','Scale':df_filter_R['Mean'].sum(),'Std':R_std})
       Rows.append({'Topic':colname, 'Party':'Democrats','Scale':df_filter_D['Mean'].sum(),'Std':D_std})
       
       """
       reps=[]
       axis=f"{colname}_D"
       for i, row in df_filter_R.iterrows():
          a=row[colname+'_R']
          reps.append(row['Version']/df_filter_D.query(f"{axis}==@a")['Version'].values[0])
     
       df_filter_R['Reps']=reps 
       df_filter_R['Topic']=[colname]*len(df_filter_R)
       name=f'{colname}_R'
       df_filter_R=df_filter_R.rename(columns={name:'Attribute'})
       """

       #Rows.append(df_filter_R)
     
    DF=pd.DataFrame.from_dict(Rows)
    #DF=DF[~(DF['Attribute'].isin(['0','9','8']))]

    return DF


def preprocess_mfq_empirical():
    ### 3 csv files named Study 1.csv , Study 2.csv, Study 3.csv 

    df = pd.read_csv('Data/MFQ_Survey_Data/Study 1.csv')
    df1=pd.read_csv('Data/MFQ_Survey_Data/Study 2.csv')
    df2=pd.read_csv('Data/MFQ_Survey_Data/Study 3.csv')

    df=df[['Political.Party','Harm', 'Fairness','Loyalty', 'Authority', 'Purity']]
    df1=df1[['1Rep.2Dem.3Other','Harm', 'Fairness','Loyalty', 'Authority', 'Purity']]
    df1=df1.rename(columns={'1Rep.2Dem.3Other':'Party'})
    df=df.rename(columns={'Political.Party':'Party'})
    df2=df2[['Party','Harm', 'Fairness','Loyalty', 'Authority', 'Purity']]

    concat_df=pd.concat([df,df1,df2],ignore_index=True)
    concat_df_mean=concat_df.groupby(['Party']).mean([['Harm', 'Fairness','Loyalty', 'Authority', 'Purity']]).reset_index()
    concat_df_mean=pd.melt(concat_df_mean,id_vars='Party',value_vars=['Harm','Fairness','Loyalty','Authority','Purity']).rename(columns={'variable':'Attribute','value':'Scale'})
    concat_df_mean=concat_df_mean.query("Party==1.0 or Party==2.0")
    concat_df_mean['Party'] = concat_df_mean['Party'].replace([1.0,2.0],['Republicans','Democrats'])
    concat_df_mean['Scale'] = concat_df_mean[['Attribute','Scale']].apply(lambda x: 7-x['Scale'] if x['Attribute']=='Fairness' or x['Attribute']=='Harm' else x['Scale'],axis=1)

    concat_df_std=concat_df.groupby(['Party']).std().reset_index()
    concat_df_std=pd.melt(concat_df_std,id_vars='Party',value_vars=['Harm','Fairness','Loyalty','Authority','Purity']).rename(columns={'variable':'Attribute','value':'Std'}) 
    concat_df_std=concat_df_std.query("Party==1.0 or Party==2.0")
    concat_df_mean['Std']=concat_df_std['Std'].tolist()

    concat_df_mean['Model']=['Empirical']*len(concat_df_mean)
    

    return concat_df_mean


def get_mfq_responses():
    Results = [] 

    model_map={'llama2_70b':'Llama2-70b','gpt-3.5':'Gpt-3.5','gpt-4':'Gpt-4','gemini':'Gemini'}
    models = ['llama2_70b','gpt-3.5','gpt-4','gemini']

    for model in models: 
        print(f'Process...{model}')
        processed = 'result/processed/'
        task = 'MFQ'
        filelists=os.listdir(processed)
        files=[f for f in filelists if f.startswith(task)]
        files=[f for f in files if model in f]

        responses = []
        for file in files:
            dff=pd.read_csv(processed+file) 
            responses.append(dff)    
        responses=pd.concat(responses,ignore_index=True)

        if model =='bard':
            mask=responses['Scale']=='[]'
            responses=responses[~mask]
            responses['Scale']=responses['Scale'].apply(lambda x: ast.literal_eval(x))
            responses['Scale']=responses['Scale'].apply(lambda x: [float(t) for t in x])
            responses['Scale'] = responses['Scale'].apply(lambda x: mean(x))

        #responses['Scale']=responses['Scale'].astype(int)
        std=responses[['Party','Attribute','Scale']].groupby(['Party','Attribute']).std().reset_index()['Scale'].tolist()
        
        responses=responses.groupby(['Party','Attribute']).mean(['Scale']).reset_index()
        
        responses['Scale'] = responses[['Attribute','Scale']].apply(lambda x: 7-x['Scale'] if x['Attribute']=='Fair' or x['Attribute']=='Harm' else x['Scale'],axis=1)
        responses=responses[['Party','Attribute','Scale']]

        responses['Std']=std
    
        responses['Model']=[model_map[model]]*len(responses)
        responses['Attribute']=responses['Attribute'].replace(['Fair','InGroup'],['Fairness','Loyalty'])
        Results.append(responses)
    responses=pd.concat(Results,ignore_index=True)
    return responses

def solve_mfq_kernel_of_truth(df):
    Rows = []
    for attribute in df['Attribute'].unique():

        EXP = df.query("Party=='Republicans' and Attribute==@attribute and Model=='Empirical'")['Scale'].values[0]
        EXM = df.query("Party=='Democrats' and Attribute==@attribute and Model=='Empirical'")['Scale'].values[0]

        for model in df['Model'].unique():
            if model=='Empirical':
                continue 
            EBP = df.query("Party=='Republicans' and Model==@model and Attribute==@attribute")['Scale'].values[0]
            r = symbols('r')
            expr = (1+r)*EXP - (r*EXM) - EBP 
            sol = solve(expr)

            print(f"{model}:: {attribute}:: {sol}")
            Rows.append({'Model':model,'Attribute':attribute,'Gamma':sol[0]})
    result=pd.DataFrame.from_dict(Rows)
    return result 

def solve_anes_kernel_of_truth(df):
    Rows = [] 
    for attribute in df['Topic'].unique():
        if attribute=='Aid_Black' and model=='Gemini':
            continue
        EXP = df.query("Party=='Republicans' and Topic==@attribute and Model=='Empirical'")['Scale'].values[0]
        EXM = df.query("Party=='Democrats' and Topic==@attribute and Model=='Empirical'")['Scale'].values[0]

        for model in df['Model'].unique():
            if model=='Empirical':
                continue 
            if attribute=='Aid_Black' and model=='Gemini':
                continue
            EBP = df.query("Party=='Republicans' and Model==@model and Topic==@attribute")['Scale'].values[0]
            r = symbols('r')
            expr = (1+r)*EXP - (r*EXM) - EBP 
            sol = solve(expr)

            print(f"{model}:: {attribute}:: {sol}")
            Rows.append({'Model':model,'Topic':attribute,'Gamma':sol[0]})
    result=pd.DataFrame.from_dict(Rows)
    return result

def count_representative(df,cols):
    Rows = [] 

    for colname in cols: 
       df[colname+'_D']=df[colname+'_D'].astype('string')
       df[colname+'_R']=df[colname+'_R'].astype('string')

       df_filter_D = df.dropna(subset=[colname+'_D'])
       mask = df_filter_D[colname+'_D']==' '
       df_filter_D=df_filter_D[~mask]

       df_filter_R = df.dropna(subset=[colname+'_R'])
       mask = df_filter_R[colname+'_R']==' '
       df_filter_R=df_filter_R[~mask]

       if colname=='Gov_services':
            df_filter_D[colname+'_D']=df_filter_D[colname+'_D'].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])
            df_filter_R[colname+'_R']=df_filter_R[colname+'_R'].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])

       if colname=='Abortion':
            df_filter_D[colname+'_D']=df_filter_D[colname+'_D'].replace(['1','2','3','4'],['4','3','2','1'])
            df_filter_R[colname+'_R']=df_filter_R[colname+'_R'].replace(['1','2','3','4'],['4','3','2','1'])

       df_filter_R=df_filter_R.groupby([colname+'_R']).count().reset_index()[[colname+'_R','Version']]
       #df_filter_R['Normalized']=df_filter_R['Version']/
       df_filter_D=df_filter_D.groupby([colname+'_D']).count().reset_index()[[colname+'_D','Version']]

       reps=[]
       axis=f"{colname}_D"
       for i, row in df_filter_R.iterrows():
          a=row[colname+'_R']
          reps.append(row['Version']/df_filter_D.query(f"{axis}==@a")['Version'].values[0])
     
       df_filter_R['Reps']=reps 
       df_filter_R['Topic']=[colname]*len(df_filter_R)
       name=f'{colname}_R'
       df_filter_R=df_filter_R.rename(columns={name:'Attribute'})

       Rows.append(df_filter_R)
     
    DF=pd.concat(Rows,ignore_index=True)
    DF=DF[~(DF['Attribute'].isin(['0','9','8']))]

    return DF

def get_p_a_X(df,cols):

    Rows = [] 
    for colname in cols: 
        df[colname]=df[colname].astype('string')
        ### Pre-process #### 
        df_filtered = df.dropna(subset=[colname])
        mask = df_filtered[colname]==' '
        df_filtered=df_filtered[~mask]
        filtered=df_filtered.groupby(['PARTY',colname]).count().reset_index()[['PARTY',colname,'Version']].query("PARTY==1 or PARTY==3")
        filtered['PARTY']=filtered['PARTY'].replace([1,3],['Democrats','Republicans'])
        #filtered['PARTY']=filtered['PARTY'].astype('string')
        filtered=filtered[~(filtered[colname].isin(['0','9']))]
        if colname=='Gov_services':
            filtered[colname]=filtered[colname].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])
        if colname=='Abortion':
            filtered[colname]=filtered[colname].replace(['1','2','3','4'],['4','3','2','1'])
        filtered['Normalized']=filtered['Version']/filtered.groupby('PARTY')['Version'].transform('sum')

        new = filtered.sort_values(['PARTY','Normalized'],ascending=False)
        new=new.rename(columns={colname:'Attribute'})
        new['Topic']=[colname]*len(new)
        Rows.append(new)
    DF=pd.concat(Rows,ignore_index=True)
    return DF

def get_representativeness_frac(df,cols):

    Rows = [] 
    for colname in cols: 
        df[colname]=df[colname].astype('string')
        ### Pre-process #### 
        df_filtered = df.dropna(subset=[colname])
        mask = df_filtered[colname]==' '
        df_filtered=df_filtered[~mask]
        filtered=df_filtered.groupby(['PARTY',colname]).count().reset_index()[['PARTY',colname,'Version']].query("PARTY==1 or PARTY==3")
        filtered['PARTY']=filtered['PARTY'].replace([1,3],['Democrats','Republicans'])
        #filtered['PARTY']=filtered['PARTY'].astype('string')
        filtered=filtered[~(filtered[colname].isin(['0','9']))]
        if colname=='Gov_services':
            filtered[colname]=filtered[colname].replace(['1','2','3','4','5','6','7'],['7','6','5','4','3','2','1'])
        if colname=='Abortion':
            filtered[colname]=filtered[colname].replace(['1','2','3','4'],['4','3','2','1'])
        filtered['Normalized']=filtered['Version']/filtered.groupby('PARTY')['Version'].transform('sum')

        #new = filtered.sort_values(['PARTY','Normalized'],ascending=False).groupby('PARTY').head(topn)
        new=filtered.set_index('PARTY').groupby(colname).apply(lambda x: x['Normalized'][-1]/x['Normalized'][0]).reset_index()
        new=new.rename(columns={colname:'Attribute',0:'Rep_frac'})
        new['Topic']=[colname]*len(new)
        Rows.append(new)
    DF=pd.concat(Rows,ignore_index=True)
    return DF

def get_MFQ_p_a_X(df): 
    Rows=[]

    cols =['Harm','Fairness','Loyalty','Authority','Purity']
    for col in cols:
        for ind in [1,2,3,4,5,6]:
            colname=f"{col}_{ind}"
            df[colname]=df[colname].astype('string')
            df_filtered = df.dropna(subset=[colname])
            mask = df_filtered[colname]==' '
            df_filtered=df_filtered[~mask]
            filtered=df_filtered.groupby(['Party',colname]).size().reset_index()
            filtered['Normalized']=filtered[0]/filtered.groupby('Party')[0].transform('sum')
            filtered=filtered.sort_values(['Party','Normalized'],ascending=False)
            filtered=filtered.rename(columns={colname:'Attribute'})
            filtered['Topic'] = [colname]*len(filtered)
            Rows.append(filtered)
    DF=pd.concat(Rows,ignore_index=True)
    return DF

def get_MFQ_representativeness_frac(df):
    Rows = [] 
    cols =['Harm','Fairness','Loyalty','Authority','Purity']
    for col in cols:
        for ind in [1,2,3,4,5,6]:
            colname = f"{col}_{ind}"
            df[colname]=df[colname].astype('string')
            ### Pre-process #### 
            df_filtered = df.dropna(subset=[colname])
            mask = df_filtered[colname]==' '
            df_filtered=df_filtered[~mask]
            filtered=df_filtered.groupby(['Party',colname]).size().reset_index()
            filtered['Normalized']=filtered[0]/filtered.groupby('Party')[0].transform('sum')
            new=filtered.set_index('Party').groupby(colname).apply(lambda x: x['Normalized'][-1]/x['Normalized'][0]).reset_index()
            new=new.rename(columns={colname:'Attribute',0:'Rep_frac'})
            new['Topic']=[colname]*len(new)
            Rows.append(new)
    DF=pd.concat(Rows,ignore_index=True)
    return DF

def solve_anes_representative(Responses,Empirical,PAN):
    ROWs=[]
    for topic in PAN['Topic'].unique(): 
        print(topic)
        pan = PAN.query("Topic==@topic")['P_An'].values[0]
        
        for model in Responses['Model'].unique():
            if topic=='Aid_Black' and model=='Gemini':
                continue
            exbr=Responses.query("Model==@model and Topic==@topic and Party=='Republicans'")['Scale'].values[0]
            exbd=Responses.query("Model==@model and Topic==@topic and Party=='Democrats'")['Scale'].values[0]

            emp_mean_r = Empirical.query("Party=='Republicans' and Topic==@topic")['Scale'].values[0]
            emp_mean_d = Empirical.query("Party=='Democrats' and Topic==@topic")['Scale'].values[0]

            exr = symbols('exr')
            exd = symbols('exd')

            expr1 = emp_mean_r + exr*(pan-1)-exbr
            expr2 = emp_mean_d - exd*(pan-1)-exbd

            sol1=solve(expr1)
            sol2=solve(expr2)           

            print(f"{model}::: Republicans:: {sol1}, Democrats:{sol2}") 
            ROWs.append({'Model':model,'Topic':topic,'Republicans':sol1[0],'Democrats':sol2[0]})
    result=pd.DataFrame.from_dict(ROWs)
    return result 

rename_cols = {
    '1Rep.2Dem.3Other':'Party',
    'Political.Party':'Party',
    '1Rep.2Dem.3Other':'Party',
    'Whether or not someone cared for someone weak or\rvulnerable':'Whether or not someone cared for someone weak or vulnerable',
    'Compassion for those who are suffering is the\rmost crucial virtue.':'Compassion for those who are suffering is the most crucial / virtue.',
    'One of the worst things a person could do is hurt a defenseless animal.':'One of the worst /  things a person could do is hurt a defenseless /  animal.',
    'Whether or not some people were treated\rdifferently than others':'Whether or not some people were treated differently than / others',
    'Whether or not someone was denied his or her\rrights':'Whether or not someone was denied his or her rights',
    'When the government makes laws, the number one\rprinciple should be ensuring that everyone is treated fairly.':'When the government makes laws, the number one principle should be / ensuring that everyone is trea...',
    'Justice is the most important requirement for a\rsociety.':'Justice is the most important requirement for a society.',
    "I think it's morally wrong that rich children\rinherit a lot of money while poor children inherit nothing.":"I think it's morally wrong that rich children inherit a lot of / money while poor children inherit...",
    'I think its morally wrong that rich children\rinherit a lot of money while poor children inherit nothing.':"I think it's morally wrong that rich children inherit a lot of / money while poor children inherit...",

    "Whether or not someone's action showed love for his\ror her country":"Whether or not someone's action showed love for his or her / country",
    'Whether or not someones action showed love for his\ror her country':"Whether or not someone's action showed love for his or her / country",
    'Whether or not someone did something to betray\rhis or her group':'Whether or not someone did something to betray his or her / group',
    "I am proud of my countrys history.":"I am proud of my country's history.",
    'People should be loyal to their family members,\reven when they have done something wrong.':'People should be loyal to their family members, even when they have / done something wrong.',
    'It is more important to be a team player than to\rexpress oneself.':'It is more important to be a team player than to express / oneself.',
    "I am proud of my countrys history.":"I am proud of my country's history.",


    'Whether or not someone showed a lack of respect\rfor authority':'Whether or not someone showed a lack of respect for / authority',
    'Whether or not someone conformed to the\rtraditions of society':'Whether or not someone conformed to the traditions of / society',
    'Whether or not an action caused chaos or\rdisorder':'Whether or not an action caused chaos or disorder',
    'Respect for authority is something all children\rneed to learn.':'Respect for authority is something all children need to / learn.',           
    'Men and women each have different roles to play\rin society.':'Men and women each have different roles to play in society.',
    "If I were a soldier and disagreed with my commanding officer's orders, I would obey anyway because that is my duty.":"If I were a soldier /  and disagreed with my commanding officer's orders, I would obey /  anyway beca...",
    "If I were a soldier and disagreed with my commanding officers orders, I would obey anyway because that is my duty.":"If I were a soldier /  and disagreed with my commanding officer's orders, I would obey /  anyway beca...",
    
    'Whether or not someone violated standards of\rpurity and decency':'Whether or not someone violated standards of purity and / decency',
    'Whether or not someone did something disgusting':'Whether or not someone did something disgusting',
    'Whether or not someone acted in a way that God would\rapprove of':'Whether or not someone acted in a way that God would approve / of',
    'People should not do things that are disgusting,\reven if no one is harmed.':'People should not do things that are disgusting, even if no one is / harmed.',
    'I would call some acts wrong on the grounds that they are unnatural.':'I / would call some acts wrong on the grounds that they are / unnatural.',
}

COLS_=[
    'Party',
    'Whether or not someone suffered emotionally',
         'Whether or not someone cared for someone weak or vulnerable',
         'Whether or not someone was cruel',
         'Compassion for those who are suffering is the most crucial / virtue.',
         'One of the worst /  things a person could do is hurt a defenseless /  animal.',
         'It can never be right to kill a human being.',
    'Whether or not some people were treated differently than / others',
             'Whether or not someone acted unfairly',
             'Whether or not someone was denied his or her rights',
             'When the government makes laws, the number one principle should be / ensuring that everyone is trea...',
             'Justice is the most important requirement for a society.',
             "I think it's morally wrong that rich children inherit a lot of / money while poor children inherit...",
     "Whether or not someone's action showed love for his or her / country",
            'Whether or not someone did something to betray his or her / group',
            "Whether or not someone showed a lack of loyalty",
            "I am proud of my country's history.",
            'People should be loyal to their family members, even when they have / done something wrong.',
            'It is more important to be a team player than to express / oneself.',
     'Whether or not someone showed a lack of respect for / authority',
              'Whether or not someone conformed to the traditions of / society',
              'Whether or not an action caused chaos or disorder',
              'Respect for authority is something all children need to / learn.',
              'Men and women each have different roles to play in society.',
              "If I were a soldier /  and disagreed with my commanding officer's orders, I would obey /  anyway beca...",
     'Whether or not someone violated standards of purity and / decency',
           'Whether or not someone did something disgusting',
           'Whether or not someone acted in a way that God would approve / of',
           'People should not do things that are disgusting, even if no one is / harmed.',
           'I / would call some acts wrong on the grounds that they are / unnatural.',
           'Chastity is an important and valuable virtue.'    
]

COLS_1={
    'Whether or not someone suffered emotionally':'Harm_1',
         'Whether or not someone cared for someone weak or vulnerable':'Harm_2',
         'Whether or not someone was cruel':'Harm_3',
         'Compassion for those who are suffering is the most crucial / virtue.':'Harm_4',
         'One of the worst /  things a person could do is hurt a defenseless /  animal.':'Harm_5',
         'It can never be right to kill a human being.':'Harm_6',
    'Whether or not some people were treated differently than / others':'Fairness_1',
             'Whether or not someone acted unfairly':'Fairness_2',
             'Whether or not someone was denied his or her rights':'Fairness_3',
             'When the government makes laws, the number one principle should be / ensuring that everyone is trea...':'Fairness_4',
             'Justice is the most important requirement for a society.':'Fairness_5',
             "I think it's morally wrong that rich children inherit a lot of / money while poor children inherit...":'Fairness_6',
     "Whether or not someone's action showed love for his or her / country":'Loyalty_1',
            'Whether or not someone did something to betray his or her / group':'Loyalty_2',
            "Whether or not someone showed a lack of loyalty":'Loyalty_3',
            "I am proud of my country's history.":'Loyalty_4',
            'People should be loyal to their family members, even when they have / done something wrong.':'Loyalty_5',
            'It is more important to be a team player than to express / oneself.':'Loyalty_6',
     'Whether or not someone showed a lack of respect for / authority':'Authority_1',
              'Whether or not someone conformed to the traditions of / society':'Authority_2',
              'Whether or not an action caused chaos or disorder':'Authority_3',
              'Respect for authority is something all children need to / learn.':'Authority_4',
              'Men and women each have different roles to play in society.':'Authority_5',
              "If I were a soldier /  and disagreed with my commanding officer's orders, I would obey /  anyway beca...":'Authority_6',
     'Whether or not someone violated standards of purity and / decency':'Purity_1',
           'Whether or not someone did something disgusting':'Purity_2',
           'Whether or not someone acted in a way that God would approve / of':'Purity_3',
           'People should not do things that are disgusting, even if no one is / harmed.':'Purity_4',
           'I / would call some acts wrong on the grounds that they are / unnatural.':'Purity_5',
           'Chastity is an important and valuable virtue.':'Purity_6' 
}



def process_mfq_for_representative():
    df = pd.read_csv('Data/MFQ_Survey_Data/Study 1.csv')
    df=df.rename(columns=rename_cols)
    df=df[COLS_]
    df1=pd.read_csv('Data/MFQ_Survey_Data/Study 2.csv')
    df1=df1.rename(columns=rename_cols)
    df2=pd.read_csv('Data/MFQ_Survey_Data/Study 3.csv')
    df2=df2.rename(columns=rename_cols)
    df1=df1[COLS_]
    df2=df2[COLS_]
    concat_df = pd.concat([df,df1,df2],ignore_index=True)
    concat_df=concat_df.rename(columns=COLS_1)
    concat_df['Party'] = concat_df['Party'].replace([1.0,2.0],['Republicans','Democrats'])
    for col in ['Harm','Fairness']:
        for i in [1,2,3,4,5,6]:
            colname =f"{col}_{i}"
            concat_df[colname]=concat_df[colname].apply(lambda x: 7-x)
    concat_df=concat_df.query("Party=='Republicans' or Party=='Democrats'")
    return concat_df

def solve_mfq_representative(Responses, Empirical, PAN):
    Row =[]
    for topic in PAN['Topic'].unique():
        print(topic)
        pan = PAN.query("Topic==@topic")['P_An'].values[0]
        
        for model in Responses['Model'].unique():
            attr=topic.split('_')[0]
            exbr=Responses.query("Model==@model and Attribute==@attr and Party=='Republicans'")['Scale'].values[0]
            exbd=Responses.query("Model==@model and Attribute==@attr and Party=='Democrats'")['Scale'].values[0]

            emp_mean_r = Empirical.query("Party=='Republicans'")[topic].values[0]
            emp_mean_d = Empirical.query("Party=='Democrats'")[topic].values[0]

            exr = symbols('exr')
            exd = symbols('exd')

            expr1 = emp_mean_r + exr*(pan-1)-exbr
            expr2 = emp_mean_d - exd*(pan-1)-exbd

            sol1 = solve(expr1)
            sol2 = solve(expr2)

            print(f"{model}::: Republicans:: {sol1}, Democrats:{sol2}") 
            Row.append({'Topic':topic,'Model':model,'Republicans':sol1[0],'Democrats':sol2[0]})
    
    result_df=pd.DataFrame.from_dict(Row)
    return result_df