
#
class Pet:
    def __init__(self, name=None, age=None):
        self.name = name
        self._age = age

    age = property()

    @age.getter
    def age(self):
        print(f'{self.name} age is {self._age}')

    @age.setter
    def age(self, age):
        if age >= 0:
            print(f'Now {self.name} age is {age}')
            self._age = age

    @age.deleter
    def age(self):
        print(f'Deleting {self.name} age')
        del self._age

    def killer(self):
        print(f'Why u kill {self.name}')

class Dog(Pet):
    def __init__(self, name, breed=None):
        super().__init__(name)
        self.breed = breed

    def say(self):
        return f'{self.name}: \'waw!\''

    age = property()
    @age.setter
    def age(self, age):
        print(f'Теперь нашей псине {self.name} исполнилось {age} лет!')
        self._age = age
dog = Dog('Саня', 'Просто Саня')
cat = Pet('Коля')
dog.age = 20
cat.age = 5
print(dog.say())
dog.killer()
