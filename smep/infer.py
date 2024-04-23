import io
import json
import boto3
import os
from datetime import datetime
import time 

REGION = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = REGION
boto3_session=boto3.session.Session()
smr = boto3.client('sagemaker-runtime')
# Change the value to reflect endpoint name in your env
endpoint_name = 'inf2-Meta-Llama-3-8B-Instruct-2024-04-23-20-19-07-642' 

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
        
def run_infer_streaming(endpoint_name, body):
    resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name,
                                                    Body=body,
                                                    ContentType="application/json")
    event_stream = resp['Body'] 
    parser = Parser()
    results = ''
    ## following solution is workaround for scan_lines() function in Parser()
    ## scan_lines() function should be adjusted to account for output JSON of llama3
    for event in event_stream:
        parser.write(event['PayloadPart']['Bytes']) 
        for line in parser.scan_lines():
            line = line.decode("utf-8")
            if len(line) < 2:
                pass
            else:
                line = line.split(":")[1]
                line = line.replace('"', '')
                if len(line) > 0:
                    line = line.strip()
                    results = f"""{results} {line}"""
    return results

body = """Could you remind me when was the C programming language was invented?""".encode('utf-8')

start = time.time()
results = run_infer_streaming(endpoint_name, body)
end = time.time()
print(f'\nPrediction took {end-start} seconds\n')
print(results)


