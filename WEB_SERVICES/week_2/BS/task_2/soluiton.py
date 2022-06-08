from bs4 import BeautifulSoup
from collections import deque
import re
import os

#   Оно банально нормально не считается. Это решение тоже
"""

def opener(path):
    with open(path, 'r', encoding='utf-8') as file:
        output = file.read()
    return output

print(file_names)
#%%

def parse(path_to_file):
    text = opener(path_to_file)
    soup = BeautifulSoup(text, 'lxml').find('div', id='bodyContent')
    print(soup.find_all('a'))

parse(f'../wiki/{file_names[0]}')
#%%
graph = dict()
def create_graph_dict_node(path_to_file, name):
    pattern = re.compile("/wiki/([\w()]+)")
    link_matched = list()
    text = opener(path_to_file)
    soup = BeautifulSoup(text, 'lxml').find('div', id='bodyContent')
    for _ in soup.find_all('a', href=True):
        _name_ = _['href'].split('/')[-1]
        if (_name_ in file_names) and (_name_ != name):
            try:
                link_matched.append(pattern.findall(_['href'])[0])
            except IndexError:
                pass
    graph.update({f"{name}": list(set(link_matched))})


for i, obj in enumerate(file_names):
    clear_output(wait=True)
    display(str(round(i/len(file_names)*100, 2)) + '%')
    create_graph_dict_node(f'../wiki/{obj}', obj)
graph
#%%
def shortest_way(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not start in graph.keys():
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = shortest_way(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


shortest_way(graph=graph, start='The_New_York_Times', end='Stone_Age')

#%%

graph2 = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}

shortest_way(graph2, 'A', 'D')
#%%

def dfs(adj, v, parent, order):
    if not parent:
        parent[v] = None
    # checking neighbours of v
    for n in adj[v]:
        if n not in parent:
            parent[n] = v
            dfs(adj, n, parent, order)

    # we're done visiting a node only when we're done visiting
    # all of its descendents first
    order.append(v)


def topological_sort(adj):
    parent = {}
    order = []
    for v in adj.keys():
        if v not in parent:
            parent[v] = None
            dfs(adj, v, parent, order)

    return list(reversed(order))

def dag_shortest_path(adj, source, dest):
    order = topological_sort(adj)
    parent = {source: None}
    d = {source: 0}

    for u in order:
        if u not in d: continue  # get to the source node
        if u == dest: break
        for v, weight in adj[u]:
            if v not in d or d[v] > d[u] + weight:
                d[v] = d[u] + weight
                parent[v] = u

    return parent, d

dag_shortest_path(graph, 'The_New_York_Times', 'Stone_Age')
#%%
from collections import deque
def find_all_paths(graph, start, end, path=[]):
    dist = {start: [start]}
    q = deque(start)
    at = q
    print(at)
    for next in graph[at]:
        if next not in dist:
            dist[next] = [dist[at], next]
            q.append(next)
    return dist.get(end)

find_all_paths(graph, 'The_New_York_Times', 'Stone_Age')
#%%
'''     FASTER
text = opener('../wiki/Echinozoa')
soup = BeautifulSoup(text, 'lxml').find('div', id='bodyContent')
for _ in soup.find_all('a', href=re.compile("/wiki/([\w()]+)")):
    print(_['href'].split('/')[-1])
'''
#%%


"""
def build_tree(start, end, path):
    link_re = re.compile(r"(?<=/wiki/)[\w()]+")
    files = dict.fromkeys(os.listdir(path))

    for file in files:
        with open(path+file, 'r') as f:
            files[file] = []
            res = link_re.findall(f.read())
            for name in res:
                if name in files and name not in files[file] and name != file:
                    files[file].append(name)

    search_queue = deque()
    search_queue += files[start]
    file_names = {start: None}

    while search_queue:
        parent = search_queue.popleft()
        for file in files[parent]:
            if file not in file_names:
                search_queue += files[parent]
                file_names[file] = parent

    return file_names


def build_bridge(start, end, path):
    files = build_tree(start, end, path)
    bridge = [end]
    # TODO Добавить нужные страницы в bridge
    file = end
    while file is not start:
        file = files[file]
        if file is None:
            break
        else:
            bridge.append(file)

    return list(reversed(bridge))


def get_count_imgs(body):
    imgs = body.findAll('img')
    widths = []
    for img in imgs:
        try:
            img = int(img['width'])
            if img > 199:
                widths.append(img)
        except KeyError:
            pass
    return len(widths)


def get_count_headers(body):
    headers = body.find_all(re.compile('^h[1-6]$'))
    count = 0
    for header in headers:
        text = header.text
        if text.startswith('E') or text.startswith('C') or text.startswith('T'):
            count += 1

    return count


def get_links_len(body):
    linkslen = 0
    for a in body.find_all('a'):
        a_siblings = a.find_next_siblings()
        len_arr = 1
        for sib in a_siblings:
            if sib.name == 'a':
                len_arr += 1
            else:
                break
        if len_arr > linkslen:
            linkslen = len_arr

    return linkslen


def get_count_lists(body):
    lists = 0
    for ul in body.find_all('ul'):
        if ul.parent.name == 'div' or ul.parent.name == "td":
            lists += 1
    for ol in body.find_all('ol'):
        if ol.parent.name == 'div' or ul.parent.name == "td":
            lists += 1

    return lists


def parse(start, end, path):


    bridge = build_bridge(start, end, path)
    out = {}
    for file in bridge:
        with open("{}{}".format(path, file)) as data:
            soup = BeautifulSoup(data, "lxml")
        body = soup.find(id="bodyContent")

        imgs = get_count_imgs(body)
        headers = get_count_headers(body)
        linkslen = get_links_len(body)
        lists = get_count_lists(body)

        out[file] = [imgs, headers, linkslen, lists]
    return out