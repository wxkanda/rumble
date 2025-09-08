# jupyter notebook

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyMuPDFLoader
import os
import concurrent.futures
import requests
import json
import time
import numpy as np
import pandas as pd
import openpyxl
from tabulate import tabulate
from langchain.schema import Document

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

### **Functions Declaration**
# Prod java
url = "https://lges2elpafunc2.azurewebsites.net/api/esg_slm?code=key"
headers = {
    "Authorization": "Bearer token",
    "Content-Type": "application/json"
}

def inference_fromjs(prompt_new):
    # Request Body
    payload = {
        "model": "finetuned/ESG-LLM2",
        "messages": [
            {"role": "user", "content": prompt_new}
        ],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 15000
    }

    # Convert to JSON format
    json_payload = json.dumps(payload)

    response = requests.post(url, headers=headers, data=json_payload)
    response=response.json()["choices"][0]["message"]["content"]
    return response
# Helper: Check if a row is fully empty or has mostly empty cells
def is_separator_row(row, threshold=0.95):
    non_empty = row.count()
    return non_empty / len(row) < (1 - threshold)

def table_extraction(df_raw):
    # Find start/end row indices for table blocks
    table_blocks = []
    start_idx = None

    for i in range(len(df_raw)):
        if not is_separator_row(df_raw.iloc[i]):
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                table_blocks.append((start_idx, i))
                start_idx = None

    # Handle last block (if the sheet ends without an empty row)
    if start_idx is not None:
        table_blocks.append((start_idx, len(df_raw)))

    # Extract tables
    tables = []
    for start, end in table_blocks:
        table = df_raw.iloc[start:end].reset_index(drop=True)
        
        # Optional: promote first row as header
        table.columns = table.iloc[0]
        table = table[1:].reset_index(drop=True)
        table=table.dropna(axis=1, how='all')   # Drop the column that is full empty in a table
        tables.append(table)
        
    return tables

#Note this method is written by considering the X-axis and Y-axis as reference. Hence the first column will always considerd as refernce point.
def data_body_prepare(df2,structured_data = []):
    # Extract the first column as row names
    row_names = df2.iloc[:, 0]
    # Remaining columns are headings
    headings = df2.columns[1:]
    # Convert to list of dicts
    table_name=df2.columns[0]
    for i in range(len(df2)):
        row_name = row_names[i]
        for heading in headings:
            value = df2.iloc[i][heading]
            structured_data.append({
                'table name': table_name,
                'heading': heading,
                'row_name': row_name,
                'value': value
            })
    return structured_data

def table_classification(tables):
    # Checking the validity of a table. A valid table should not have single row or single column
    invalid_tables=[]
    valid_table=[]
    garbage=[]
    for table in tables:
        m,n=table.shape
        if m<1 or n<=1:
            if m>=1 and n>=1:
                invalid_tables.append(table)
            else:
                garbage.append(table)
        else:
            valid_table.append(table)

    # Preparing the contaxt to embed as table informaiton
    valid_table_context=[]
    for index_val in range(0,len(valid_table)):
        col_value=valid_table[index_val].columns.to_list()
        row_value=valid_table[index_val].iloc[:,0].to_list()
        table_name=col_value[0]
        col_value=col_value[1:]
        context="Table name: "+str(table_name)
        result = ', '.join(str(x) for x in col_value)
        context=context+";\n Heading: "+ result
        result = ', '.join(str(x) for x in row_value)
        context=context+";\n Information: "+ result
        valid_table_context.append(context)

    return valid_table,valid_table_context,invalid_tables,garbage

def count_words(sentence):
    words = sentence.split()
    return len(words)

# This format will save the table name, heading
def data_body_prepare_v3(df2,structured_data = {}):
    clean_data={}
    table_name=df2.columns[0]
    row_names = df2.iloc[:, 0]
    headings = df2.columns[1:]
    first_row_names=list(row_names.values)
    heading_name=headings.to_list()
    for head_namei in heading_name:
        clean_head_data={}
        clean_head_data['heading']=head_namei
        for i in range(len(df2)):
            # print("Heading name:",head_namei,'; row name:',first_row_names[i],'; Value:',df2.iloc[i][head_namei])
            row_name=first_row_names[i]
            value=df2.iloc[i][head_namei]
            cleaned_row = {
                'row_name': row_name,
                'value': value
            }
            clean_head_data.setdefault('Row information', []).append(cleaned_row)  
        clean_data.setdefault(table_name, []).append(clean_head_data) 

    key_name=list(clean_data.keys())
    table_saved_name=key_name[0]
    structured_data[table_saved_name]=clean_data[table_saved_name]
    return structured_data


def clean_value(value):
    if isinstance(value, (np.generic, np.ndarray)):
        return value.item()
    return value  # Strings and native types are fine

def data_body_prepare_v31(df2,structured_data = {}):
    clean_data={}
    table_name=df2.columns[0]
    row_names = df2.iloc[:, 0]
    headings = df2.columns[1:]
    first_row_names=list(row_names.values)
    heading_name=headings.to_list()
    for head_namei in heading_name:
        clean_head_data={}
        clean_head_data['heading']=clean_value(head_namei)
        clean_head_data['table_name']=clean_value(table_name)
        for i in range(len(df2)):
            # print("Heading name:",head_namei,'; row name:',first_row_names[i],'; Value:',df2.iloc[i][head_namei])
            row_name=first_row_names[i]
            value=df2.iloc[i][head_namei]
            cleaned_row = {
                'row_name': clean_value(row_name),
                'value': clean_value(value)
            }
            clean_head_data.setdefault('Row information', []).append(cleaned_row)  
        clean_data.setdefault(table_name, []).append(clean_head_data) 

    key_name=list(clean_data.keys())
    table_saved_name=key_name[0]
    structured_data[table_saved_name]=clean_data[table_saved_name]
    return structured_data

def sample41(user_query,intermediate_answer,justification):
    prompt_test = f"""
        You are a helpful assistant. Your task is to evaluate whether the provided `intermediate_answer` accurately and completely addresses the `user_query`. Then, generate a final `output_response` that correctly responds to the `user_query` in a clear, pleasant, and helpful tone.

        You will be provided with:
        - `user_query`
        - `intermediate_answer`
        - `justification`

        Rules:
        1. If the `intermediate_answer` or `justification` are based on the assumptions, then respond with: "Information not found."
        2. If the `intermediate_answer` does not contain relevant, correct, or complete information required to answer the `user_query`, respond with: "Information not found."
        3. If the `intermediate_answer` explicitly states that the information cannot be determined or is not available or not sufficient to answer user quesry or doing some assumption to answer the user query, then respond with: "Information not found."
        4. If the `user_query` explicitly requests a tabular format, and the `intermediate_answer` is following rule 1, 2, or 3, then return the answer in a readable table.
        5. Do not add any extra information that was not asked for. Avoid giving explanations beyond what is requested.

        **Inputs:**
        - user_query: {user_query}
        - intermediate_answer: {intermediate_answer}
        - justification: {justification}

        **Output:**
        output_response:
            """
    return prompt_test

def sample42(user_query,intermediate_answer,table):
    prompt_test = f"""
    You are a helpful assistant. Your task is to evaluate whether the provided `intermediate_answer` accurately and completely addresses the `user_query`. Then, generate a final `output_response` that correctly responds to the `user_query` in a clear, pleasant, and helpful tone.

    You will be provided with:
    - `user_query`
    - `intermediate_answer`
    - `json_table` for the answer

    **Rules:**
    1. Ensure that all values in the `intermediate_answer` are derived from or match entries in the `json_table`. If values are incorrect or misspelled, correct them.
    2. Do not include any information related to the "Qwen" model, as it was used for fine-tuning.
    3. If the `intermediate_answer` merely repeats the query or provides irrelevant or fabricated information, respond with: **"Information not found."**
    4. Validate the values provided in the 'intermediate_answer' is present in the `json_table`, if it is not matching then correct it.

    **Inputs:**
    - user_query: {user_query}
    - intermediate_answer: {intermediate_answer}
    - json_table: {table}

    **Output:**
    output_response:
    """
    return prompt_test

def sample_prompt_gen_simple_v3(user_query, json_file):
    prompt_test = f"""
    You are a data analytics expert. Your task is to analyze the following JSON data and answer the user's query.

    **Instructions:**
    - Respond only to the user query.
    - Do not include any additional commentary or explanation.
    - Your output must be a valid JSON object with the following keys:
    - "answer": A concise and accurate response to the user query.
    - "justification": A brief explanation of how the answer was derived.
        - If any assumptions were made (e.g., interpreting ambiguous terms or inferring relationships not explicitly stated in the data), clearly state that it is an assumption and explain the reasoning.
    - "table_name": A list of table names used to derive the answer.

    **Input JSON Data:**
    {json_file}

    **User Query:**
    {user_query}

    **Expected Output Format:**
    {{
    "answer": "...",
    "justification": "...",
    "table_name": ["..."]
    }}
    """
    return prompt_test

def sample_prompt_gen_simple_v31(user_query, json_file):
    prompt_test = f"""
    You are a data analytics expert. Your task is to analyze the following JSON data and answer the user's query.

    **Instructions:**
    - Respond only to the user query.
    - Do not include any additional commentary or explanation.
    - Do not perform any mathematical calculations or aggregations yourself.
    - If the query requires mathematical operations, provide Python code to perform them under the key "python_code".
    - Your output must be a valid JSON object with the following keys:
        - "answer": A concise response to the user query using raw values from the data (no calculations or aggregations).
        - "justification": A brief explanation of how the answer was derived.
            - If any assumptions were made (e.g., interpreting ambiguous terms or inferring relationships not explicitly stated in the data), clearly state that it is an assumption and explain the reasoning.
        - "table_name": A list of table names used to derive the answer.
        - "python_code": Python code to perform any required calculations (if applicable).

    **Important:**
    - Do not compute totals, averages, percentages, or any other derived values in the "answer" field.
    - Example of correct behavior:
        - Query: "What is the total number of employees?"
        - Answer: "22459 (Female), 25560 (Male)"
        - Python Code: "total_employees = 22459 + 25560"

    **Input JSON Data:**
    {json_file}

    **User Query:**
    {user_query}

    **Expected Output Format:**
    {{
        "answer": "...",
        "justification": "...",
        "table_name": ["..."],
        "python_code": "..."
    }}
    """
    return prompt_test

def prompt_find_table(user_query,table_information):
	prompt_test = f"""
	You are given a `user_query` and a list of `table_information` in a dictionary. Each item in `table_information` contains:
	- `table_name`: Name of the table
	- `heading`: A brief description of the table's columns
	- `information`: A description of what each row in the table represents

	Your task is to identify which tables are required to answer the `user_query`. The user query may require information from multiple tables. Return the indices (0-based) of the relevant tables in a JSON format like this:
	{{
	"table_index": [0, 2, 4, 5, 6]
	}}

	**Important Instructions:**
	- Prefer tables that contain **direct, summarized, or final answers** (e.g., averages, totals, percentages).
	- Give **highest importance to the `information` field**, as it describes what each row represents.
	- Also consider `heading` and `table_name` to understand the table's relevance.
	- If fewer than 5 tables are clearly relevant, include additional tables that may be **partially helpful** or **contextually related**, up to a minimum of 5 total.
	- Do **not** include any explanation or extra text—only return the JSON output.

	### Example Input:
	user_query: "Show me the sales performance of each product category by region."

	table_information:
	[
	0: "table_name: ProductSales;\n heading: ProductID, Category, Region, SalesAmount; \n information: Each row represents the sales of a product in a specific region.",
	1: "table_name: CustomerFeedback; \n heading: ProductID, FeedbackScore, Comments \n information: Each row represents customer feedback for a product.",
	2: "table_name: InventoryLevels; \n heading: ProductID, Warehouse, StockLevel; \n information: Each row represents the stock level of a product in a warehouse."
	]

	### Expected Output:
	{{
	"table_index": [0, 1, 2]
	}}

	**Inputs:**
	- user_query: {user_query}
	- table_information: {table_information}

	**Output:**
	output_response:
	"""
	return prompt_test


file_path = r"C:\Users\k.shesha.naik\Downloads\2024-NAB-sustainability-data-pack_sample for excel as input_vetted (3).xlsx"
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names
print("Sheet present")
print(sheet_names)

required_sheet=[]
cnt_table=0
documents = []
valid_main_table=[]
Valid_table_sheet_name=[]
valid_main_context=[]
valid_main_context_dict={}
for sheet_name_point in sheet_names:
    # print("sheet:",i)
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name_point, engine='openpyxl')
    tables=table_extraction(df_raw)
    valid_table,valid_table_context,invalid_tables,garbage=table_classification(tables)
    print("sheet:",sheet_name_point,'; Valid table:',len(valid_table),'; invalid_tables:',len(invalid_tables),'; garbage:',len(garbage))
    # if cnt_table>2:
    #     break
    # Prepare the data into VectorDB
    debug_tables=[]
    for valid_table_index, chunk in enumerate(valid_table_context):
        valid_main_table.append(valid_table[valid_table_index])
        debug_tables.append(valid_table[valid_table_index].columns[0])
        Valid_table_sheet_name.append(sheet_name_point)
        valid_main_context.append(valid_table_context[valid_table_index])
        valid_main_context_dict[cnt_table]=valid_table_context[valid_table_index]
        doc = Document(
            page_content=chunk,
            metadata={
                "source":cnt_table,  # dummy unique identifier
            }
        )
        documents.append(doc)
        cnt_table=cnt_table+1   
    print("Table loaded:",cnt_table,": valid table used:",debug_tables) 


print("Number of valid main table context:",len(valid_main_context))

file_path = r"C:\Users\k.shesha.naik\Downloads\ESRS_S1 E5 E6_Topics_Prompt_Instructions_Quant_Multiple sheet-testing_inference.xlsx"
df_query = pd.read_excel(file_path, engine='openpyxl')
df_query['combined_query']=df_query['KPI QUERY']+"\n user instruction:\n"+df_query['PROMPT INSTRUCTIONS']
user_query_test_list=df_query['combined_query']

response_list_1={}
response_list_2={}
inference_capture=[]
justification_capture=[]
table_name_capture=[]

for cnt in range(0,len(user_query_test_list)):
    # print("\n Query: ",user_query_test_list[cnt])
    flag=''
    user_query=user_query_test_list[cnt]

    # # Vector search
    # results_with_scores = vectorstore.similarity_search_with_relevance_scores(user_query, k=10)

    # short_listed_table_index=[]
    # for doc, score in results_with_scores:
    #     short_listed_table_index.append(doc.metadata['source'])
        
    # Short listing of table using the llm    
    find_tables=prompt_find_table(user_query,valid_main_context_dict)
    llm_output=inference_fromjs(find_tables)  
    result_dict = json.loads(llm_output)
    short_listed_table_index=result_dict['table_index'][0:10]  # will consider first 10 tables only
    # print(result_dict['table_index'])
    flag=flag+'; find_table llmcall'

    structured_data = {}
    for i in short_listed_table_index:
        structured_data=data_body_prepare_v31(valid_main_table[i],structured_data = structured_data)

    Json_file = json.dumps(structured_data, indent=2)
    main_prompt=sample_prompt_gen_simple_v31(user_query,Json_file)
    answer_1=inference_fromjs(main_prompt)                          # 1st SLM call
    json_data = json.loads(answer_1)
    flag=flag+'; 1st slm infer'

    justification_capture.append(json_data['justification'])
    table_name_capture.append(json_data['table_name'])
    if count_words(str(json_data['answer']))>10:
        keys_to_extract=json_data['table_name']
        subset_dict = {key: structured_data[key] for key in keys_to_extract if key in structured_data}
        validation_prompt_1=sample41(user_query, json_data['answer'],json_data['justification'])
        answer_2=inference_fromjs(validation_prompt_1)   
        flag=flag+'; 2nd slm infer'                 # 2nd SLM call
        if answer_2 != 'Information not found.' and answer_2 != 'Information not found' and answer_2.upper()!='INFORMATION NOT FOUND.' and answer_2.upper()!='INFORMATION NOT FOUND':
            Json_file_sub = json.dumps(subset_dict, indent=2)
            validation_prompt2=sample42(user_query, answer_2,Json_file_sub)
            answer_2=inference_fromjs(validation_prompt2)                # 3rd SLM call
        flag=flag+'; 3rd slm infer'
    else:
        answer_2=json_data['answer']

    print('cnt:',cnt,'; ',flag)
    time.sleep(10)
    inference_capture.append(answer_2)
    break

df_query=pd.DataFrame()
df_query['User Query']=user_query_test_list
df_query['Infernce testing']=inference_capture
df_query['Justification']=justification_capture
df_query['Tables refered']=table_name_capture

df_query.to_csv('testing_infer7.csv', index=False)
