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
    "value": 12,
    "result": "variable is a multiple of two"
  },
  {
    "id": 2,
    "value": 9,
    "result": "variable is a multiple of three"
  },
  {
    "id": 3,
    "value": 25,
    "result": "variable is a multiple of five"
  },
  {
    "id": 4,
    "value": 13,
    "result": "variable is not a multiple of 2, 3 and 5"
  }
]

for test in tests:
  if (solution(test['value']) == test['result']):
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

task_id = '1.2.4'
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