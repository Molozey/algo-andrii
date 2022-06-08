class SomeObject:
    def __init__(self):
        self.integer_field = 0
        self.float_field = 0.0
        self.string_field = ""

FINTEGER, FFLOAT, FSTRING = "INT", "FLOAT", "STR"

class EventGet:
    def __init__(self, value):
        self.kind = {int: FINTEGER, float: FFLOAT, str: FSTRING}[value]
        self.value = None


class EventSet:
    def __init__(self, value):
        self.kind = {int: FINTEGER, float: FFLOAT, str: FSTRING}[type(value)]
        self.value = value


class NullHandler:
    def __init__(self, successor=None):
        self.__successor = successor

    def handle(self, obj : SomeObject, event):
        if self.__successor is not None:
            return self.__successor.handle(obj, event)

class IntHandler(NullHandler):
    def handle(self, obj: SomeObject, event):
        if event.kind == FINTEGER:
            if event.value is None:
                return obj.integer_field
            else:
                obj.integer_field = event.value
        else:
            return super().handle(obj, event)


class FloatHandler(NullHandler):
    def handle(self, obj: SomeObject, event):
        if event.kind == FFLOAT:
            if event.value is None:
                return obj.float_field
            else:
                obj.float_field = event.value
        else:
            return super().handle(obj, event)


class StrHandler(NullHandler):
    def handle(self, obj: SomeObject, event):
        if event.kind == FSTRING:
            if event.value is None:
                return obj.string_field
            else:
                obj.string_field = event.value
        else:
            return super().handle(obj, event)


