import urllib.request, json
api_key = open(r'C:\Users\Brian\.openclaw\zai_api_key.txt').read().strip()
for base in ['https://api.z.ai', 'https://api.z.ai/v1', 'https://api.z.ai/api']:
    try:
        url = base + '/models'
        headers = {'Authorization': f'Bearer {api_key}'}
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        data = result.get('data', [])
        print(f'{base}: OK - {len(data)} models')
        for m in data[:5]:
            print(f'  {m["id"]}')
        break
    except Exception as e:
        print(f'{base}: {e}')
