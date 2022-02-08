from abc import ABC, abstractmethod
from pandas import DataFrame
from Constructor.knotslinkers import BaseInputKnot, BaseOutputKnot
from Constructor.objectTypes import Frame, Extractor, OpenGraph, CloseGraph, StrategyHub


class BaseGraphsAndHubs(ABC):
    @abstractmethod
    def __init__(self):
        self.AllInputs = list()
        self.AllOutputs = list()

    def create_parent_link(self, self_out_position, childNode, child_in_position) -> None:
        """
        Вызывает создание связи родитель-потомок
        :param self_out_position: Позиция родительского узла в ноде
        :param childNode: Дочерняя узел в дочерней ноде
        :param child_in_position: Позиция дочернего узла в дочерней ноде
        :return: None
        """
        self.AllOutputs[self_out_position]['knot'].add_child_link(childNode.AllInputs[child_in_position]['knot'])


    def remove_parent_link(self, self_out_position, childNode, child_in_position) -> None:
        """
        Вызывает удваление связи родитель-потомок
        :param self_out_position: Позиция родительского узла в ноде
        :param childNode: Дочерняя узел в дочерней ноде
        :param child_in_position: Позиция дочернего узла в дочерней ноде
        :return: None
        """
        self.AllOutputs[self_out_position]['knot'].remove_child_link(childNode.AllInputs[child_in_position]['knot'])


class BaseStrategyHub(BaseGraphsAndHubs):

    def __init__(self):
        super().__init__()
        self._extractor = None

        self._left_border = None
        self._right_border = None
        self._inside_all_data = None

        self.LookBack_period = None     # Can be taken by OpenGraph attribute (Takes like max(STAT_LOOKING))
        self.PotentialHold_period = None    # Can be taken by CloseGraph attribute

        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(Frame())},  # Семпл передающийся через экстрактор
                          {'position': 2, 'knot': BaseInputKnot(Extractor())},  # Связь хаба с экстрактором
                          {'position': 3, 'knot': BaseInputKnot(OpenGraph())},  # Связь хаба с графом открытия позиции
                          {'position': 4, 'knot': BaseInputKnot(CloseGraph())}]  # Связь хаба с графом удержания позиции

        self.AllOutputs = [{'position': 1, 'knot': BaseOutputKnot(StrategyHub())},    #   Передача хаба в графы
                           {'position': 2, 'knot': BaseOutputKnot(Frame())},    # Передача актуальных данных в экстрактор
                           {'position': 3, 'knot': BaseOutputKnot(Frame())},    # Передача данных полученных с экстрактора в граф открытия
                           {'position': 4, 'knot': BaseOutputKnot(Frame())}]    # Передача данных полученных с экстрактора в граф удержания

        self.AllOutputs[0]['knot'].changed_condition(self)

    @property
    def get_left_border(self):
        return self._left_border

    @property
    def get_right_border(self):
        return self._right_border

    def transport_start_data(self, raw_data: DataFrame):
        if type(raw_data) != DataFrame:
            raise TypeError('Wrong Input data')

        self._inside_all_data = raw_data.copy()


class BaseStrategyExtractor(BaseStrategyHub):

    def __init__(self):
        super().__init__()
        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(StrategyHub())},
                          {'position': 2, 'knot': BaseInputKnot(Frame())}]

        self.AllOutputs = [{'position': 1, 'knot': BaseOutputKnot(Extractor())},  # Отвечает за передачу экстрактора
                           ]

        self.AllOutputs[0]['knot'].changed_condition(self)

        self.return_batch = None

    def create_parent_link(self, self_out_position, childNode, child_in_position) -> None:
        """
        Вызывает создание связи родитель-потомок
        :param self_out_position: Позиция родительского узла в ноде
        :param childNode: Дочерняя узел в дочерней ноде
        :param child_in_position: Позиция дочернего узла в дочерней ноде
        :return: None
        """
        self.AllOutputs[self_out_position]['knot'].add_child_link(childNode.AllInputs[child_in_position]['knot'])


    def remove_parent_link(self, self_out_position, childNode, child_in_position) -> None:
        """
        Вызывает удваление связи родитель-потомок
        :param self_out_position: Позиция родительского узла в ноде
        :param childNode: Дочерняя узел в дочерней ноде
        :param child_in_position: Позиция дочернего узла в дочерней ноде
        :return: None
        """
        self.AllOutputs[self_out_position]['knot'].remove_child_link(childNode.AllInputs[child_in_position]['knot'])

    def test_calculation(self):
        self.return_batch = [1, 2, 3, 4, 5]
