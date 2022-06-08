#!/usr/bin/python3

import json
import requests
import os
# from app import solution

result = {}

with open('result.txt', 'r') as f:
  result['code'] = f.read()

with open('app.py', 'r') as f:
  app_test = f.read()
  result['app'] = True
  words = ['52022', 'paramiko', 'from_private_key_file', '/root/pwd.txt', 'cat']
  for word in words:
    if word not in app_test:
      result['app'] = False

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

task_id = '2.2.4'
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