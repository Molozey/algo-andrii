from bs4 import BeautifulSoup






def parse(path_to_file):
    AVAILABLE_START_CHARS = ['E', 'T', 'C']

    def opener(path):
        with open(path, 'r', encoding='utf-8') as file:
            output = file.read()
        return output

    def widthlogical(obj):
        WIDTH_CONST = 200
        try:
            if int(obj['width']) >= WIDTH_CONST:
                return True
            else:
                return False
        except:
            return False
text = opener(path_to_file)
soup = BeautifulSoup(text, 'lxml')
soup = soup.find('div', id='bodyContent')
    imgs = len(list(filter(widthlogical, soup.find_all('img'))))
    headers = len((list(filter(lambda x: x.text[0] in AVAILABLE_START_CHARS, soup.find_all([f'h{i}' for i in range(1, 7)])))))

    """    parent_source = soup.find_all('a')
    init_parent = parent_source[0]
    init_len = 1
    buffer_len = 1
    for i, obj in enumerate(parent_source[1:]):
        if obj.parent == init_parent:
            buffer_len += 1
        if obj.parent != init_parent:
            init_parent = obj.parent
            print(parent_source[i-buffer_len+1:i+1])
            if buffer_len > init_len:
                init_len = buffer_len
            buffer_len = 1"""
    init_len = 0
    parent_source = soup.find_all('a')
    for link in parent_source:
        count = 1
        next_links = link.find_next_siblings()
        for sublink in next_links:
            if sublink.name == 'a':
                count += 1
                init_len = max(init_len, count)
            else:
                count = 0

    list_sum = 0
    all_lists = soup.find_all(['ul', 'ol'])
    for tag in all_lists:
        if not tag.find_parents(['ul', 'ol']):
            list_sum += 1
    return [imgs, headers, init_len, list_sum]


#print(parse('wiki/Ingo_Rechenberg'))