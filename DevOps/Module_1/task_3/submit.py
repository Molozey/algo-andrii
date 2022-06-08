#!/usr/bin/python3

import json
import requests
import os
from app import solution

result = []

message_mask  = 'With value ({}) test should return ({}) but returned ({})'

tests = [
  {
    "id": 1,
    "value": '12',
    "result": {'status': 'OK', 'value': 12}
  },
  {
    "id": 2,
    "value": '0',
    "result": {'status': 'OK', 'value': 0}
  },
  {
    "id": 3,
    "value": 'twenty five',
    "result": {'status': 'ERR', 'value': "This string cannot be converted to a number"}
  },
  {
    "id": 4,
    "value": [12],
    "result": {'status': 'ERR', 'value': "You passed a value that is not a string"}
  }
]

for test in tests:
  run_result = None
  try:
    run_result = solution(test['value'])
  except ValueError:
    if test['result'] == "This string cannot be converted to a number":
      result.append({
          "status": True
        })
    else:
      result.append({
        "status": False,
        "message": message_mask.format(test['value'], test['result'], 'ValueError')
      })
    continue
  except TypeError:
    if test['result'] == "You passed a value that is not a string":
      result.append({
          "status": True
        })
    else:
      result.append({
        "status": False,
        "message": message_mask.format(test['value'], test['result'], 'TypeError')
      })
    continue


  if run_result == test['result']:
    result.append({
      "status": True
    })
  else:
    result.append({
      "status": False,
      "message": message_mask.format(test['value'], test['result'], solution(test['value']))
    })


coursera = {
  '1.2.4': {
    'key': 'YduLJZbST2O11cNO02J7TQ',
    'part': 'OEZu0'
  },
  '1.3.3': {
    'key': 't-6J7CYuQGqnYMdutqprgw',
    'part': 'xtYm7'
  },
  '1.4.3': {
    'key': 'n1yievvAQ0yTdtZAioj2yA',
    'part': 'dCcMZ'
  },
  '2.2.4': {
    'key': 'AM-1JMltSj64NI7_AG2XZA',
    'part': 'Zh8o0'
  },
  '3.2.3': {
    'key': '8QE6TK_cRfui5yFa_XtKag',
    'part': 'ktOPa'
  },
  '4.1.4': {
    'key': 'YyaNNddLTba18km7v4Hy-A',
    'part': 'ag1Gn'
  }
}

task_id = '1.4.3'
email = input('Set your email:')
coursera_token = input('Set coursera token:')

submission = {
  "assignmentKey": coursera[task_id]['key'],
  "submitterEmail":  email,
  "secret":  coursera_token,
  "parts": {
    coursera[task_id]['part']: {"output": json.dumps(result)}
  }
}

response = requests.post('https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1', data=json.dumps(submission))

if response.status_code == 201:
  print ("Submission successful, please check on the coursera grader page for the status")
else:
  print ("Something went wrong, please have a look at the reponse of the grader")
  print ("-------------------------")
  print (response.text)
  print ("-------------------------")