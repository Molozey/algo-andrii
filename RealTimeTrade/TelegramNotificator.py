
import telebot
from threading import Thread
from telebot import types
import datetime
import time

global LastUpdate
LastUpdate = datetime.datetime.now()
user_dict = {'249910303': 'Andrii'}


class TelegramNotification:
    def __init__(self):
        self.TOKEN_SAVER = None

        API_TOKEN = '5331091759:AAExx0cPRkGehwxPDg9LvLqhi1HBW1PDopY'
        self.bot = telebot.TeleBot(API_TOKEN)
        del API_TOKEN

        @self.bot.message_handler(content_types='text')
        def message_reply(message, inside=self):
            inside.bot.send_message(message.chat.id, 'Saved Token')
            inside.TOKEN_SAVER = message.text
            print('TKN', bot.TOKEN_SAVER)
            bot.bot.stop_polling()
            print('Stop polling')

    def make_message(self):
        for chatId, name in user_dict.items():
            print('Send message to:', name)
            self.bot.send_message(chatId, 'Please send a message with token')
            self.bot.polling(True)
        print('send all messages')
        # self.bot.stop_polling()
        print('Polled')



def useRobot(bot):
    LastUpdate = datetime.datetime.now()
    while True:
        time.sleep(10)
        if (datetime.datetime.now() - LastUpdate) > datetime.timedelta(seconds=60):
            print('Past token', bot.TOKEN_SAVER)
            LastUpdate = datetime.datetime.now()
            bot.make_message()
            print('New token', bot.TOKEN_SAVER)
        print('Now token', bot.TOKEN_SAVER)

# bot.bot.stop_bot()
bot = TelegramNotification()
th = Thread(target=useRobot, args=(bot,)).start()
def printer():
    while True:
        time.sleep(5)
        print('Curent time:', datetime.datetime.now().minute)
thPrinter = Thread(target=printer, args=()).start()
# useRobot(bot)