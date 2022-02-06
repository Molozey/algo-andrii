#%%

from abc import ABC, abstractmethod
from Constructor.objectTypes import Value, Frame as DataFrame
from Constructor.abstractNode import BaseNode
from Constructor.knotslinkers import BaseInputKnot, BaseOutputKnot


class ImportDataNode(BaseNode):
    priority = 0

    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = []
        self.AllOutputs = [{'position': 1, 'knot': BaseOutputKnot(DataFrame())}]

        self._inside = None

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

        self._inside = read_csv(filepath_or_buffer=filepath, sep=sep, header=header, index_col=index_col)
        self.AllOutputs[0]['knot'].changed_condition(self._inside)


    def execute_node(self):
        pass

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

        self._inside = None

    def execute_node(self):
        pass

    def node_cycle(self, **param):
        self._inside = self.AllInputs[0]['knot'].getContent()

        self.AllOutputs[0]["knot"].changed_condition(float(self._inside.open.max()))
        pass

class TestingPrintNode(BaseNode):
    priority = 0
    def __init__(self):
        super().__init__(priority=self.priority)
        self.AllInputs = [{'position': 1, 'knot': BaseInputKnot(Value())}]
        self.AllOutputs = []

        self._inside = None

    def execute_node(self):
        pass

    def node_cycle(self, **param):
        print(self.AllInputs[0]['knot'].getContent())
        pass

    def NodeIndex(self):
        return 'HelperNode_001'


testNode = TestingPrintNode()
dataImport = ImportDataNode()
calculateMax = CalculateMax()
calculateMax.create_parent_link(self_inp_position=0, parentNode=dataImport, parent_out_position=0)
testNode.create_parent_link(self_inp_position=0, parentNode=calculateMax, parent_out_position=0)
# dataImport.create_parent_link(self_out_position=0, childNode=calculateMax, child_in_position=0)
# calculateMax.create_parent_link(self_out_position=0, childNode=testNode, child_in_position=0)

dataImport.node_cycle(filepath='test.csv', sep=',', header=0, index_col=0)
calculateMax.node_cycle()
testNode.node_cycle()