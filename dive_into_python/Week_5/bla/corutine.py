def grep(pattern):
    print('Start grep')
    try:
        while True:
            input_data = yield
            if pattern in input_data:
                print(input_data)
    except GeneratorExit:
        print('Stop grep')

pattern_check = grep('ABOBA')
next(pattern_check)
pattern_check.send('BOBAABOBA')
pattern_check.throw(RuntimeError, 'something wrong')
pattern_check.close()
