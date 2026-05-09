import requests
response = requests.get("https://httpbin.org/get?name=Tom")
print(response.json())