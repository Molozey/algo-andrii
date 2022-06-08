from abc import ABC, abstractmethod


class ObservableEngine(Engine):
    """Обертка над движком, позволяет подписывать наблюдателей и рассылать им уведомления"""
    def __init__(self):
        self.__subscribers = set()

    def subscribe(self, subscriber):
        self.__subscribers.add(subscriber)

    def unsubscribe(self, subscriber):
        self.__subscribers.remove(subscriber)

    def notify(self, new_achievement):
        for SUB in self.__subscribers:
            SUB.update(new_achievement)


class AbstractObserver(ABC):
    """Абстракция для наблюдателя"""
    @abstractmethod
    def update(self, ach):
        pass


class ShortNotificationPrinter(AbstractObserver):
    def __init__(self):
        self.achievements = set()

    def update(self, achievement):
        self.achievements.add(achievement["title"])


class FullNotificationPrinter(AbstractObserver):
    def __init__(self):
        self.achievements = list()

    def update(self, achievement):
        if achievement not in self.achievements:
            self.achievements.append(achievement)
