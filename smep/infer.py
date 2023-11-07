import io
import json
import boto3
import os
from datetime import datetime
import time 

boto3_session=boto3.session.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
    aws_session_token=os.environ['AWS_SESSION_TOKEN'],
    region_name="us-east-1")
smr = boto3.client('sagemaker-runtime')
# Change the value to reflect endpoint name in your env
endpoint_name = 'smep-inf2-llama2-7b-chat-2023-11-01-22-04-58-025' 

class Parser:
    """
    A helper class for parsing the byte stream input. 
    
    The output of the model will be in the following format:
    ```
    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...
    ```
    
    While usually each PayloadPart event from the event stream will contain a byte array 
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}
    ```
    
    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character) within
    the buffer via the 'scan_lines' function. It maintains the position of the last read 
    position to ensure that previous bytes are not exposed again. 
    """
    
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0
        
    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)
        data = self.buff.getvalue()
        
    def scan_lines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b'\n':
                self.read_pos += len(line)
                yield line[:-1]
                
    def reset(self):
        self.read_pos = 0
        
def run_infer(endpoint_name, body):
    resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name,
                                                    Body=body,
                                                    ContentType="application/json")
    event_stream = resp['Body'] 
    parser = Parser()
    results = ''
    for event in event_stream:
        parser.write(event['PayloadPart']['Bytes']) 
        for line in parser.scan_lines():
            #print(line.decode("utf-8"), end=' ')
            results = results + line.decode("utf-8") + ' '
    return results

body = """User: Explain the self-attention mechanism that Transformers use like I'm ten years old.
          Assistant:""".encode('utf-8')
start = time.time()
results = run_infer(endpoint_name, body)
end = time.time()
print(f'\nPrediction took {end-start} seconds\n')
#print(f'\nThis is the result of inference request #1 \n\n {results}')

body2 = """
    Write a concise summary of the text, return your responses with 2 lines that cover
the key points of the following text.
    ```
    Intended Use
    Intended Use Cases Llama 2 is intended for commercial and research use in English.
    Tuned models are intended for assistant-like chat, whereas pretrained models can
be
    adapted for a variety of natural language generation tasks.
    To get the expected features and performance for the chat versions, a
    specific formatting needs to be followed, including the INST and <<SYS>> tags,
    BOS and EOS tokens, and the whitespaces and breaklines in between (
        we recommend calling strip() on inputs to avoid double-spaces).
        See our reference code in github for details: chat_completion.
    Out-of-scope Uses Use in any manner that violates applicable laws or regulations
    (including trade compliance laws).Use in languages other than English.
    Use in any other way that is prohibited by the Acceptable Use Policy and Licensing
Agreement for Llama 2.
    ```
    SUMMARY:
    """.encode('utf-8')
start = time.time()
results = run_infer(endpoint_name, body2)
end = time.time()
print(f'\nPrediction took {end-start} seconds\n')

#print(f'\nThis is the response for warm up inference request #2:\n\n {results}\n')
