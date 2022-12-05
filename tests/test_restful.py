import json
import requests

url = 'http://127.0.0.1:5000/label/upload'
# url = 'http://127.0.0.1:5000/todata/api/v1.0/tasks'
headers = {"content-type": "application/json"}

d = {
    'id': 3,
    'title': u'Buy groceries',
    'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
    'done': False,
    'image_path': 'c:\\pro\\r\\2022.jpg'
}

# r = requests.post(url, data=json.dumps(d), headers=headers)

url = 'http://127.0.0.1:5000/label/2022.json'
r = requests.get(url)
print(r.text)
