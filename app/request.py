import requests
data = [{
    "MCQ_A10":0,
    "MCQ_A11":1,
    "MCQ_A13":2,
    "MCQ_A14":3,
    "MCQ_A15":4,
    "MCQ_A18":5,
    "MCQ_A2":0,
    "MCQ_A20":1,
    "MCQ_B7":2,
    "MCQ_B76":3,
    "MCQ_B78":4,
    "MCQ_B79":5,
    "MCQ_B8":0,
    "MCQ_B82":2,
    "MCQ_B85":3,
    "MCQ_B89":5,
    "MCQ_B90":2,
    "MCQ_B91":1,
    "MCQ_B95":2,
    "MCQ_B98":3,
    "text_response":"INTP roommate about what you said and he says Hi",
    }]

url = "http://127.0.0.1:5000/api"
response = requests.post(url,json=data)
json_response = response.text
print(response.status_code)
print(json_response)