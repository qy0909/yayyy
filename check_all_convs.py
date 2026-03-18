import requests
import json

# Get conversations
response = requests.get('http://localhost:8000/api/conversations')
data = response.json()

print(f'Total conversations: {len(data.get("conversations", []))}')
print('\nChecking each conversation for sources:\n')

for conv in data.get('conversations', [])[:3]:  # Check first 3
    conv_id = conv['id']
    title = conv.get('title', 'Unknown')[:40]
    
    # Load the conversation
    conv_response = requests.get(f'http://localhost:8000/api/conversations/{conv_id}')
    conv_data = conv_response.json()
    
    messages = conv_data.get('messages', [])
    print(f'Conversation: {title}')
    print(f'  Messages: {len(messages)}')
    
    # Check if any message has sources
    has_sources = False
    for msg in messages:
        if msg.get('sources') and len(msg['sources']) > 0:
            has_sources = True
            print(f'  ✓ Message with {len(msg["sources"])} sources found')
            break
    
    if not has_sources:
        print(f'  ✗ No sources in any message')
        # Check what keys messages have
        if messages:
            print(f'    First message keys: {list(messages[0].keys())}')
    print()
