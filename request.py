import requests

url = 'http://localhost:5000/crop_prediction'

r = requests.post(url,json={'input':[[88,40,41,25.09865, 85.12345, 7.01010, 200.09854323]]})

print(r.json())