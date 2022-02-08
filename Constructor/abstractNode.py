from abc import ABC, abstractmethod
from knotslinkers import *
from objectTypes import *


class BaseNode(ABC):
    """
    Базовый класс для создания новых нодовых структур
    """
    def __init__(self, priority):
        """
        Создает списки из принимающих и отдающих узлов
        :param priority: Приоритет выполнения ноды. Используются в фильтрующих нодах для оптимизации времени фильтрации
        """
        self.priority = priority
        self.AllInputs = list()
        self.AllOutputs = list()
        self.buffer_condition = dict()
        self.new_outputs = list()

        self._children_nodes = list()

        self._actualOutputs = [{'position': outKnot['position'], 'content': outKnot['knot'].getContent()} for outKnot in
                               self.AllOutputs]

    def mutable_logic(self):
        if self.new_outputs is not  self._actualOutputs:
            for outKnot in self.AllOutputs:
                for newKnot in self.new_outputs:
                    if newKnot['position'] == outKnot['position']:
                        outKnot['knot'].changed_condition(newKnot['content'])

            self.execute_node()
    # def create_parent_link(self, self_inp_position, parentNode, parent_out_position) -> None:
    #     """
    #     Вызывает создание связи потомок-родитель
    #     :param self_inp_position: Позиция дочернего узла в дочерней ноде
    #     :param parentNode: Родительская нода с которой связывается узел
    #     :param parent_out_position: Позиция родительского узла в родительской ноде
    #     :return: None
    #     """
    #     parentNode.AllOutputs[parent_out_position]['knot'].add_child_link(self.AllInputs[self_inp_position]['knot'])

    def create_parent_link(self, self_out_position, childNode, child_in_position) -> None:
        """
        Вызывает создание связи родитель-потомок
        :param self_out_position: Позиция родительского узла в ноде
        :param childNode: Дочерняя узел в дочерней ноде
        :param child_in_position: Позиция дочернего узла в дочерней ноде
        :return: None
        """
        self.AllOutputs[self_out_position]['knot'].add_child_link(childNode.AllInputs[child_in_position]['knot'])

        if childNode not in self._children_nodes:
            self._children_nodes.append(childNode)

    def remove_parent_link(self, self_out_position, childNode, child_in_position) -> None:
        """
        Вызывает удваление связи родитель-потомок
        :param self_out_position: Позиция родительского узла в ноде
        :param childNode: Дочерняя узел в дочерней ноде
        :param child_in_position: Позиция дочернего узла в дочерней ноде
        :return: None
        """
        self.AllOutputs[self_out_position]['knot'].remove_child_link(childNode.AllInputs[child_in_position]['knot'])

        if childNode in self._children_nodes:
            self._children_nodes.remove(childNode)

    @abstractmethod
    def node_cycle(self, **param):
        """
        Описания процесса преобразования данных из входящих узлов в выходящие узлы
        :param param:
        :return:
        """
        pass

    def execute_node(self):
        """
        Функция запускающая принудительное выполнение ноды
        :return:
        """
        for child in self._children_nodes:
            child.node_cycle()

    @property
    @abstractmethod
    def NodeIndex(self):
        """
        У каждой ноды должен быть свой уникальный индекс. Это необходимо для возможности добавления GUI интерфейса
        :return: str с индексом ноды
        """
        pass

    def NodeName(self):
        """
        Имя ноды определяется как имя класса наследованного от базового класса
        :return: Имя ноды
        """
        return self.__class__.__name__


# Примеры НОД

class GetABSValue(BaseNode):
    priority = 3

    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(Value())},
                          {'position': 2, 'knot': BaseInputKnot(Value())}]
        self.AllOutputs = []

    def convert(self):
        return abs(self.AllInputs[0]['knot'].KnotObject.receive())

    def show_condition(self):
        for inputs in self.AllInputs:
            print(f"{inputs['position']}, condition: {inputs['knot'].KnotObject.receive()}")

    def NodeIndex(self):
        return 'G_001'


class ConstNode(BaseNode):
    priority = 3

    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = list()
        self.AllOutputs = [{'position': 1, 'knot': BaseOutputKnot(Value())}]

        self._inside = None
        self._mutable = None

    def node_cycle(self, value):
        buffer = self._inside
        self._inside = value

        if buffer != self._inside:
            self._mutable = True
        del buffer

    def execute_node(self):
        if self._mutable:
            self.AllOutputs[0]['knot'].changed_condition(self._inside)
        self._mutable = False

    def NodeIndex(self):
        return 'G_002'


# getValue = GetABSValue()
# CONST = ConstNode()
# getValue.create_parent_link(self_inp_position=0,parentNode=CONST, parent_out_position=0)
#
#
# getValue.show_condition()
