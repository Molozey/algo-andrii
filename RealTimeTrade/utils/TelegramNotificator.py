from abc import ABC, abstractmethod
import telebot
from threading import Thread
from telebot import types
import datetime
import time


user_dict = {'249910303': 'Andrii'}


class AbstractFastNotification(ABC):
    @abstractmethod
    def send_message_to_user(self, message):
        pass


class TelegramNotification(AbstractFastNotification):

    def get_stored_token(self):
        return self.TOKEN_SAVER

    @classmethod
    def token_request_echo(cls, insideBot, tradingInterface):
        while True:
            time.sleep(10)
            LastUpdate = tradingInterface.lastTokenUpdate
            # print('lastUpdateType:', type(LastUpdate))
            # print('lastUpdateValue:', LastUpdate)
            # print(datetime.datetime.now())
            # print(datetime.datetime.now() - LastUpdate)
            # print(datetime.timedelta(seconds=tradingInterface.tokenLife // 2).total_seconds())
            if (datetime.datetime.now() - LastUpdate).total_seconds() > \
                    datetime.timedelta(seconds=tradingInterface.tokenLife // 2).total_seconds():
                LastUpdate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                insideBot.make_token_request_message()
                tradingInterface.update_token_information(token=insideBot.get_stored_token(), updateDate=LastUpdate)

    def __init__(self):
        self.TOKEN_SAVER = None

        API_TOKEN = '5331091759:AAExx0cPRkGehwxPDg9LvLqhi1HBW1PDopY'
        self.bot = telebot.TeleBot(API_TOKEN)
        del API_TOKEN

        @self.bot.message_handler(content_types='text')
        def message_reply(message, inside=self):
            inside.bot.send_message(message.chat.id, 'Saved Token')
            inside.TOKEN_SAVER = message.text
            # print('TKN', self.TOKEN_SAVER)
            self.bot.stop_polling()
            # print('Stop polling')

    def make_token_request_message(self):
        for chatId, name in user_dict.items():
            # print('Send message to:', name)
            self.bot.send_message(chatId, 'Please send a message with token')
            # self.bot.polling(True)
        self.bot.polling(True)
        # print('send all messages')
        # self.bot.stop_polling() # Dont use it
        # print('Polled')

    def send_message_to_user(self, message):
        for chatId, name in user_dict.items():
            self.bot.send_message(chatId, f"""message for {name}:\n{message}""")



# def token_request_echo(bot, tradingInterface=None):
#     LastUpdate = datetime.datetime.now()
#     while True:
#         time.sleep(10)
#         if (datetime.datetime.now() - LastUpdate).total_seconds() > \
#                 datetime.timedelta(seconds=tradingInterface.tokenLife // 2).total_seconds():
#             print('Past token', bot.TOKEN_SAVER)
#             LastUpdate = datetime.datetime.now()
#             bot.make_token_request_message()
#             print('New token', bot.TOKEN_SAVER)
#         print('Now token', bot.TOKEN_SAVER)
#
# # bot.bot.stop_bot()
# if __name__ == "__main__":
#     bot = TelegramNotification()
#     th = Thread(target=token_request_echo, args=(bot,)).start()
#     def printer():
#         while True:
#             time.sleep(5)
#             print('Curent time:', datetime.datetime.now().minute)
#     thPrinter = Thread(target=printer, args=()).start()
#     # token_request_echo(bot)