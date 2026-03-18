import requests
import json

# Call the backend API with a test query
response = requests.post('http://localhost:8000/api/chat', json={
    'query': 'What are labor rights?',
    'top_k': 5,
    'conversation_id': None,
    'conversation_history': []
}, timeout=30)

if response.status_code == 200:
    data = response.json()
    print('✓ Backend API response received')
    print(f'Status: {data.get("status")}')
    print(f'Success: {data.get("success")}')
    print(f'RAG Used: {data.get("rag_used")}')
    print(f'Sources count: {len(data.get("sources", []))}')
    if data.get('sources'):
        print(f'\nFirst source keys: {list(data["sources"][0].keys())}')
        print(f'\nFirst source (full):')
        print(json.dumps(data['sources'][0], indent=2, default=str))
    else:
        print('\n⚠ No sources returned!')
        print(f'Response keys: {list(data.keys())}')
else:
    print(f'Error: {response.status_code}')
    print(response.text[:200])
