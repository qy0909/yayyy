import requests
import json

# First, get the list of conversations
response = requests.get('http://localhost:8000/api/conversations')
data = response.json()

if data.get('conversations'):
    conv_id = data['conversations'][0]['id']
    print(f'Loading conversation: {conv_id}\n')
    
    # Now load the specific conversation
    conv_response = requests.get(f'http://localhost:8000/api/conversations/{conv_id}')
    conv_data = conv_response.json()
    
    if conv_data.get('messages'):
        print(f'Total messages: {len(conv_data["messages"])}')
        print('\nLast 2 messages:')
        for i, msg in enumerate(conv_data['messages'][-2:], 1):
            print(f'\nMessage {i}:')
            print(f'  Role: {msg.get("role")}')
            print(f'  Text: {msg.get("text")[:60]}...')
            print(f'  Has sources key: {"sources" in msg}')
            if 'sources' in msg:
                print(f'  Sources count: {len(msg["sources"])}')
                if msg['sources']:
                    print(f'  First source title: {msg["sources"][0].get("title")}')
            else:
                print(f'  Message keys: {list(msg.keys())}')
    else:
        print('No messages in conversation')
else:
    print('No conversations found')
