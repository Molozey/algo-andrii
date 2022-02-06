from abc import ABC, abstractmethod


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

    def create_parent_link(self, self_inp_position, parentNode, parent_out_position) -> None:
        """
        Вызывает создание связи потомок-родитель
        :param self_inp_position: Позиция дочернего узла в дочерней ноде
        :param parentNode: Родительская нода с которой связывается узел
        :param parent_out_position: Позиция родительского узла в родительской ноде
        :return: None
        """
        parentNode.AllOutputs[parent_out_position]['knot'].add_child_link(self.AllInputs[self_inp_position]['knot'])

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
        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(Value())}, {'position': 2, 'knot': BaseInputKnot(Value())}]
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

    def change_condition(self, value):
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
