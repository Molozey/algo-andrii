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
        """
        Вызывает проверку изменения состояния выходов в ноде. В случае если изменения есть - каскадно вызываются все дочерние ноды.
        :return:
        """
        if self.new_outputs is not self._actualOutputs:
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

class ImportDataNode(BaseNode):
    priority = 0

    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = []
        self.AllOutputs = [{'position': 1, 'knot': BaseOutputKnot(DataFrame())}]

        self._actualOutputs = [{'position': outKnot['position'], 'content': outKnot['knot'].getContent()}for outKnot in self.AllOutputs]

    def node_cycle(self, **param):
        from pandas import read_csv

        get_params = param
        if not 'filepath' in get_params:
            raise KeyError('No filepath')
        else:
            filepath = param['filepath']
        if not 'sep' in get_params:
            sep = ','
        else:
            sep = param['sep']
        if not 'header' in get_params:
            header = 0
        else:
            header = param['header']
        if not 'index_col' in get_params:
            index_col = 0
        else:
            index_col = param['index_col']

        self.new_outputs = [{'position':1, 'content':read_csv(filepath_or_buffer=filepath, sep=sep, header=header, index_col=index_col)}]

        self.mutable_logic()

    def NodeIndex(self):
        return 'D_001'


class CalculateMax(BaseNode):
    priority = 0

    def NodeIndex(self):
        return 'C_001'

    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(DataFrame())}]
        self.AllOutputs = [{'position': 1, 'knot': BaseOutputKnot(Value())}]

        self.buffer_condition = [{'position': 1, 'content': self.AllOutputs[0]['knot'].getContent()}]
        self._inside = self.buffer_condition

    def node_cycle(self, **param):

        self.new_outputs = [{'position': 1, 'content': float(self.AllInputs[0]['knot'].getContent().open.max())}]

        self.mutable_logic()


class TestingPrintNode(BaseNode):
    priority = 0

    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(Value())}]
        self.AllOutputs = []

        self.buffer_condition = []
        self._inside = self.buffer_condition

    def node_cycle(self, **param):
        print(self.AllInputs[0]['knot'].getContent())
        pass

    def NodeIndex(self):
        return 'HelperNode_001'

# testNode = TestingPrintNode()
# dataImport = ImportDataNode()
# calculateMax = CalculateMax()
# dataImport.create_parent_link(self_out_position=0, childNode=calculateMax, child_in_position=0)
# calculateMax.create_parent_link(self_out_position=0, childNode=testNode, child_in_position=0)
#
#
# dataImport.node_cycle(filepath='test.csv', sep=',', header=0, index_col=0)
# dataImport.node_cycle(filepath='test2.csv')