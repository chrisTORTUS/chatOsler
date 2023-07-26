from dotenv import load_dotenv
import openai
import json

DEFAULT_SYSTEM = '''
You are a GP taking notes after your consultation with a patient. You are extracting order and medicine info from the transcript of the consultation. DO NOT include any information that is not in the transcript.
'''

with open('functions.json', 'r') as f:
    FUNC = json.load(f)

def gpt_msg(system: str, query: str):
    '''
    Creates gpt messages object.

    Params:
        - system: system message
        - query: User query
    '''
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': query}
    ]

    return messages

def parse_func_call(response):
    '''
    Parses GPT function call response.
    '''
    resp_dict = response.to_dict()
    if 'function_call' in response.to_dict():
        return json.loads(resp_dict['function_call']['arguments'])
    return resp_dict['content']

def ask_gpt(query: str, func=FUNC, system: str = DEFAULT_SYSTEM,  model_name: str = 'gpt-4-0613'):
    msgs = gpt_msg(system, query)
    completion = openai.ChatCompletion.create(
        model= model_name,
        messages = msgs,
        functions=[func],
        function_call={'name': func['name']}
    )
    
    #reply = json.loads(str(completion))
    return parse_func_call(completion.choices[0].message)