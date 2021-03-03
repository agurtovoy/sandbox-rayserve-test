import requests
import sys

file_path = sys.argv[1]
r = requests.put("http://127.0.0.1:8000/pointrend", data=open(file_path, 'rb'), headers={'Content-type': 'image/jpeg'})
print("{}: {}".format(r.status_code, r.headers['content-type'] if r.status_code == 200 else r.text))
