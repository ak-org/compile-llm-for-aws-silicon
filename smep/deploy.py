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
endpoint_name = 'smep-inf2-llama2-13b-chat-2023-10-25-19-49-33-570' #"smep-inf2-llama2-13b-b1-2023-10-20-04-53-19-441"

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
        

payload = {
    "inputs": [
        [
            {"role": "user", "content": "what is the recipe of mayonnaise?"},
        ]
    ],
    "parameters": {"max_new_tokens": 1024, "top_p": 0.9, "temperature": 0.6},
}
start = time.time()
resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name, 
                                                Body=json.dumps(payload), 
                                                ContentType="application/json")
event_stream = resp['Body']
parser = Parser()
for event in event_stream:
    parser.write(event['PayloadPart']['Bytes'])
    for line in parser.scan_lines():
        print(line.decode("utf-8"), end=' ')
print('\n')
end = time.time()
print(f'Prediction took {end-start} seconds\n')

payload2 = {
    "inputs": [
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ]
    ],
    "parameters": {"max_new_tokens": 384, "top_p": 0.9, "temperature": 0.6},
}
start = time.time()
resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name, 
                                                Body=json.dumps(payload2), 
                                                ContentType="application/json")
event_stream = resp['Body']
parser = Parser()
for event in event_stream:
    parser.write(event['PayloadPart']['Bytes'])
    for line in parser.scan_lines():
        print(line.decode("utf-8"), end=' ')
print('\n')
end = time.time()
print(f'Prediction took {end-start} seconds\n')

payload3 = payload = {
    "inputs": [
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ]
    ],
    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6},
}
start = time.time()
resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name, 
                                                Body=json.dumps(payload3), 
                                                ContentType="application/json")
event_stream = resp['Body']
parser = Parser()
for event in event_stream:
    parser.write(event['PayloadPart']['Bytes'])
    for line in parser.scan_lines():
        print(line.decode("utf-8"), end=' ')
print('\n')
end = time.time()
print(f'Prediction took {end-start} seconds\n')
