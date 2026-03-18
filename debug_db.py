import sqlite3
import json

db = sqlite3.connect('backend/conversations.db')
db.row_factory = sqlite3.Row
cursor = db.cursor()

# Check if metadata column exists
try:
    cursor.execute('SELECT metadata FROM messages LIMIT 1')
    print('✓ metadata column EXISTS')
    
    # Get recent messages with metadata
    cursor.execute('SELECT role, text, metadata FROM messages ORDER BY rowid DESC LIMIT 3')
    rows = cursor.fetchall()
    
    if rows:
        print(f'\nLast 3 messages:')
        for i, row in enumerate(rows, 1):
            print(f'\nMessage {i}:')
            print(f'  Role: {row["role"]}')
            text = row["text"]
            print(f'  Text: {text[:60]}...' if len(text) > 60 else f'  Text: {text}')
            metadata = row['metadata']
            if metadata:
                try:
                    meta_obj = json.loads(metadata)
                    print(f'  Metadata keys: {list(meta_obj.keys())}')
                    if 'sources' in meta_obj:
                        print(f'  Sources count: {len(meta_obj["sources"])}')
                        if meta_obj['sources']:
                            print(f'    First source keys: {list(meta_obj["sources"][0].keys())}')
                except Exception as e:
                    print(f'  Metadata error: {e}')
            else:
                print(f'  Metadata: (empty or None)')
    else:
        print('No messages found in database')
        
except sqlite3.OperationalError as e:
    print(f'✗ metadata column DOES NOT EXIST: {e}')
    print('Database schema is outdated. You may need to delete backend/conversations.db and restart.')
    
db.close()
