from collections import Counter
zen = """
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
"""
words_dict = dict()
for _ in zen.lower().split():
    _ = _.strip('.,!*-')
    if _ != '':
        words_dict.setdefault(_, 1)
#print(words_dict.keys())
words_tuple = []
for _ in zen.lower().split():
    _ = _.strip('.,!*-')
    words_tuple.append(_)
words_tuple = tuple(words_tuple)
for _ in words_dict.keys():
    words_dict[_] = words_tuple.count(_)
for _ in list(words_dict.items()):
    #print(f'{_[0]}: {_[1]}')
    pass

words_dict_2 = dict()
for word in zen.split():
    clean = word.strip('.,!*-').lower()
    if clean not in words_dict_2:
        words_dict_2[clean] = 0
    words_dict_2[clean] += 1
zen_items = words_dict_2.items()
#print(sorted(zen_items, key=lambda x: x[1], reverse=True))
words_list = []
for _ in zen.split():
    clean = _.strip(',.!*-').lower()
    words_list.append(clean)
print(Counter(words_list).most_common())
