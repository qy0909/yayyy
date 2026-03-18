import sqlite3
import json

db = sqlite3.connect('backend/conversations.db')
db.row_factory = sqlite3.Row
cursor = db.cursor()

# Get the latest messages
cursor.execute('SELECT role, text, metadata FROM messages ORDER BY rowid DESC LIMIT 2')
rows = cursor.fetchall()

for i, row in enumerate(rows, 1):
    print(f'Message {i}:')
    print(f'  role: {row["role"]}')
    print(f'  text: {row["text"][:40]}...')
    
    metadata_str = row['metadata']
    if metadata_str:
        print(f'  metadata length: {len(metadata_str)}')
        print(f'  metadata preview: {metadata_str[:100]}...')
        try:
            meta = json.loads(metadata_str)
            print(f'  metadata keys: {list(meta.keys())}')
            if 'sources' in meta:
                print(f'  sources count: {len(meta["sources"])}')
        except Exception as e:
            print(f'  metadata parse error: {e}')
    else:
        print(f'  metadata: (empty or None)')
    print()

db.close()
