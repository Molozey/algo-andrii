from bs4 import BeautifulSoup
import re
import os

get_links_cache = {}

def get_links(file, path, os_path_exists):
    result = []
    if file in get_links_cache: return get_links_cache[file]
    if not file: return result

    ### mine - slow
    # if not os.path.exists(os.path.join(path, file)): return result
    # with open(os.path.join(path, file), encoding='utf-8') as data:
    #     soup = BeautifulSoup(data.read(), "lxml")
    # links = soup.find(id="bodyContent").find_all('a')
    # for each in links:
    #     link = each.get('href', '')
    #     if '/wiki/' in link:
    #         link = link[link.rfind('/')+1:]
    #         if not os.path.exists(os.path.join(path, link)): continue
    #         if link not in result: result.append(link)

    ### sample
    link_re = re.compile(r"(?<=/wiki/)[\w()]+")
    with open(os.path.join(path, file), encoding='utf-8') as data:
        links = re.findall(link_re, data.read())
    for link in links:
        if link not in result and link in os_path_exists: result.append(link)

    get_links_cache[file] = result
    # print(f'{file} : {len(result)}')
    return result

def deeper(tree, level, max_level, start, end, path, os_path_exists):
    if level > max_level: return 'None'
    for each in start:
        tmp = get_links(each, path, os_path_exists)
        start[each] = dict.fromkeys(tmp)
        if end in tmp: return f'{each}|{end}'
        result = each + '|' + deeper(tree, level + 1, max_level, start[each], end, path, os_path_exists)
        if 'None' not in result: return result
    return 'None'

def build_tree(start, end, path):
    tree = {}
    os_path_exists = os.listdir('wiki/')
    if start == end: return [start]
    tree[start] = dict.fromkeys(get_links(start, path, os_path_exists))
    if end in tree[start]: return [start, end]

    max_level = 1
    while True:
        # print(i)
        result = deeper(tree, 1, max_level, tree[start], end, path, os_path_exists)
        max_level += 1
        if 'None' not in result:
            result = f'{start}|{result}'
            return result.split('|')
        # if max_level > 10: return None


def get_links(file, path, os_path_exists):
    result = []
    if file in get_links_cache: return get_links_cache[file]
    if not file: return result

    ### mine - slow
    # if not os.path.exists(os.path.join(path, file)): return result
    # with open(os.path.join(path, file), encoding='utf-8') as data:
    #     soup = BeautifulSoup(data.read(), "lxml")
    # links = soup.find(id="bodyContent").find_all('a')
    # for each in links:
    #     link = each.get('href', '')
    #     if '/wiki/' in link:
    #         link = link[link.rfind('/')+1:]
    #         if not os.path.exists(os.path.join(path, link)): continue
    #         if link not in result: result.append(link)

    ### sample
    link_re = re.compile(r"(?<=/wiki/)[\w()]+")
    with open(os.path.join(path, file), encoding='utf-8') as data:
        links = re.findall(link_re, data.read())
    for link in links:
        if link not in result and link in os_path_exists: result.append(link)

    get_links_cache[file] = result
    # print(f'{file} : {len(result)}')
    return result


def deeper(tree, level, max_level, start, end, path, os_path_exists):
    if level > max_level: return 'None'
    for each in start:
        tmp = get_links(each, path, os_path_exists)
        start[each] = dict.fromkeys(tmp)
        if end in tmp: return f'{each}|{end}'
        result = each + '|' + deeper(tree, level + 1, max_level, start[each], end, path, os_path_exists)
        if 'None' not in result: return result
    return 'None'


# Вспомогательная функция, её наличие не обязательно и не будет проверяться
def build_tree(start, end, path):
    # Искать ссылки можно как угодно, не обязательно через re
    link_re = re.compile(r"(?<=/wiki/)[\w()]+")
    # Словарь вида {"filename1": None, "filename2": None, ...}
    files = dict.fromkeys(os.listdir(path))
    # TODO Проставить всем ключам в files правильного родителя в значение, начиная от start
    for file in files:
        with open(path + '/' + file) as f:
            read_data = f.read()
        f.closed
        # Найти в файле все ссылки /wiki/, присутствующие в папке с файлами;
        # добавить их в словарь
        links = link_re.findall(read_data)
        links_for_file = []
        for link in links:
            if link in files:
                if link not in links_for_file and link != file:
                    links_for_file.append(link)
        files[file] = links_for_file
    return files


# Вспомогательная функция, её наличие не обязательно и не будет проверяться
def build_bridge(start, end, path):
    files = build_tree(start, end, path)
    bridge = []
    # TODO Добавить нужные страницы в bridge
    # Для поиска кратчайшего пути применяется алгоритм обхода графа в ширину
    # Создает очередь и помещает туда вершину, с которой необходимо начать
    queue = list([start])
    used = list([start])    # Вершины, которые уже горят
    parents = dict()        # Массив предков
    parents[start] = None
    # Пока очередь не пуста
    while len(queue) > 0:
        # Взять элемент в начале очереди
        vertex = queue.pop(0)
        # Осматривает соседей
        for neighbour in files[vertex]:
            # Если соседня вершина еще не горит
            if neighbour not in used:
                # Поджечь и поставить в очередь
                used.append(neighbour)
                queue.append(neighbour)
                parents[neighbour] = vertex
    # Определение пути от start до end
    item = end
    bridge.append(item)
    while item != start:
        item = parents[item]
        bridge.append(item)
    bridge.reverse()
    return bridge
# Вспомогательная функция, её наличие не обязательно и не будет проверяться


# Вспомогательная функция, её наличие не обязательно и не будет проверяться
def build_bridge(start, end, path):
    files = build_tree(start, end, path)
    bridge = []
    # TODO Добавить нужные страницы в bridge
    # Для поиска кратчайшего пути применяется алгоритм обхода графа в ширину
    # Создает очередь и помещает туда вершину, с которой необходимо начать
    queue = list([start])
    used = list([start])    # Вершины, которые уже горят
    parents = dict()        # Массив предков
    parents[start] = None
    # Пока очередь не пуста
    while len(queue) > 0:
        # Взять элемент в начале очереди
        vertex = queue.pop(0)
        # Осматривает соседей
        for neighbour in files[vertex]:
            # Если соседня вершина еще не горит
            if neighbour not in used:
                # Поджечь и поставить в очередь
                used.append(neighbour)
                queue.append(neighbour)
                parents[neighbour] = vertex
    # Определение пути от start до end
    item = end
    bridge.append(item)
    while item != start:
        item = parents[item]
        bridge.append(item)
    bridge.reverse()
    return bridge

def get_statistics(start, end, path):
    out = {}
    for file in build_tree(start, end, path):
        with open(os.path.join(path, file), encoding='utf-8') as data:
            soup = BeautifulSoup(data.read(), "lxml")
        body = soup.find(id="bodyContent")

        # Количество картинок (img) с шириной (width) не меньше 200
        imgs = 0
        for each in body.find_all('img'):
            width = int(each.attrs.get('width', 0))
            if width >= 200: imgs += 1

        # Количество заголовков, первая буква текста внутри которого: E, T или C
        headers = 0
        for each in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if each.text[0] in 'ETC': headers += 1

        # Длина максимальной последовательности ссылок, между которыми нет других тегов
        linkslen = 0
        linkslen_tmp = 1
        for each in body.find_all('a'):
            if each.next_sibling and each.next_sibling.next_sibling and each.next_sibling.next_sibling.name == 'a':
                linkslen_tmp += 1
                linkslen = max(linkslen, linkslen_tmp)
            else:
                linkslen_tmp = 1

        # Количество списков, не вложенных в другие списки
        lists = 0
        for each in body.find_all(['ul', 'ol']):
            tags = [tag.name for tag in each.parents]
            if 'ul' not in tags and 'ol' not in tags: lists += 1

        out[file] = [imgs, headers, linkslen, lists]

    return out
