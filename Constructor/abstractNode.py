from abc import ABC, abstractmethod


class BaseNode(ABC):

    def __init__(self, priority):
        self.priority = priority
        self.AllInputs = list()
        self.AllOutputs = list()

    def create_parent_link(self, self_inp_position, parentNode, parent_out_position):
        parentNode.AllOutputs[parent_out_position]['knot'].add_child_link(self.AllInputs[self_inp_position]['knot'])

    @property
    @abstractmethod
    def NodeIndex(self):
        pass

    def NodeName(self):
        return self.__class__.__name__

