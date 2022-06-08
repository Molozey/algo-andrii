import math
import random

import pygame
from typing import List

SCREEN_DIM = (800, 600)
X = 0
Y = 1


class Vec2d:

    def __init__(self, coordinates: tuple):
        self.coordinates = coordinates
        self.speed = random.random() * 2, random.random() * 2

    def __add__(self, other):
        """возвращает сумму двух векторов"""
        return Vec2d((self.coordinates[X] + other.coordinates[X],
                      self.coordinates[Y] + other.coordinates[Y]))

    def __sub__(self, other):
        """"возвращает разность двух векторов"""
        return Vec2d((self.coordinates[X] - other.coordinates[X],
                      self.coordinates[Y] - other.coordinates[Y]))

    def __mul__(self, other):
        """возвращает произведение вектора на число"""
        return Vec2d((self.coordinates[X] * other,
                      self.coordinates[Y] * other))

    def __len__(self):
        """возвращает длину вектора"""
        return math.sqrt(self.coordinates[X] * self.coordinates[X] +
                         self.coordinates[Y] * self.coordinates[Y])

    def int_pair(self):
        """возвращает кортеж из двух целых чисел
        (текущие координаты вектора)"""
        return int(self.coordinates[X]), int(self.coordinates[Y])


class Polyline:
    def __init__(self):
        self.points = []
        self.knot = []

    def add_point(self, current_line):
        """функция добавления опорной точки"""
        point = Vec2d(event.pos)
        self.points.append(point)
        current_line.get_knot()

    def set_points(self, current_line):
        """функция перерасчета координат опорных точек"""
        for point in self.points:
            point.coordinates = (Vec2d(point.coordinates) +
                                 Vec2d(point.speed)).int_pair()
            if point.coordinates[X] > SCREEN_DIM[X] or \
                    point.coordinates[X] < 0:
                point.speed = -point.speed[X], point.speed[Y]
            if point.coordinates[Y] > SCREEN_DIM[Y] or \
                    point.coordinates[Y] < 0:
                point.speed = point.speed[X], -point.speed[Y]
        current_line.get_knot()

    def draw_points(self, line_color):
        """функция отрисовки точек на экране"""
        width = 3

        for point in self.points:
            pygame.draw.circle(gameDisplay, (255, 255, 255),
                               point.int_pair(), width)

        if len(self.points) >= 3:
            for p_n in range(-1, len(self.knot) - 1):
                pygame.draw.line(gameDisplay, line_color,
                                 self.knot[p_n].int_pair(),
                                 self.knot[p_n + 1].int_pair(), width)

    @staticmethod
    def draw_help(polylines: List):
        """функция отрисовки экрана справки программы"""
        gameDisplay.fill((50, 50, 50))
        font1 = pygame.font.SysFont("courier", 24)
        font2 = pygame.font.SysFont("serif", 24)

        data = [
            ["F1", "Show Help"],
            ["R", "Restart"],
            ["P", "Pause/Play"],
            ["Num+", "More points"],
            ["Num-", "Less points"],
            ["S", "Speed higher"],
            ["L", "Speed lower"],
            ["D", "Delete last point"],
            ["A", "Append new line"],
            ["", ""]
        ]
        for i in range(len(polylines)):
            data.append([f"Line {i + 1}",
                         f"{polylines[i].count} current points"])

        pygame.draw.lines(gameDisplay, (255, 50, 50, 255), True, [
            (0, 0), (800, 0), (800, 600), (0, 600)], 5)
        for i, text in enumerate(data):
            gameDisplay.blit(font1.render(
                text[0], True, (128, 128, 255)), (100, 100 + 30 * i))
            gameDisplay.blit(font2.render(
                text[1], True, (128, 128, 255)), (200, 100 + 30 * i))


class Knot(Polyline):
    """
    данный класс содержит функции, отвечающие за расчет сглаживания прямой
    """

    def __init__(self):
        super().__init__()
        self.count = 35

    def _get_point(self, base_points: List, alpha: float, deg: int = None):
        if deg is None:
            deg = len(base_points) - 1
        if deg == 0:
            return base_points[0]
        return base_points[deg] * alpha + self._get_point(
            base_points, alpha, deg - 1) * (1 - alpha)

    def _get_points(self, base_points: List) -> List:
        alpha = 1 / self.count
        res = []
        for i in range(self.count):
            res.append(self._get_point(base_points, i * alpha))
        return res

    def get_knot(self) -> List:
        """
        данная функция отвечает за расчёт точек кривой по добавляемым
        «опорным» точкам
        """
        SCALE = 0.5

        self.knot = []

        if len(self.points) >= 3:
            for i in range(-2, len(self.points) - 2):
                base_points = [(self.points[i] + self.points[i + 1]) *
                               SCALE, self.points[i + 1],
                               (self.points[i + 1] + self.points[i + 2]) *
                               SCALE]
                self.knot.extend(self._get_points(base_points))
        return self.knot


def coefficients(polylines: List, flag: bool = None):
    """
    Функция для расчета новой скорости движения точек. Чтобы для случая
    уменьшения скорости скорость на стала отрицательной, был введен
    коэффициент изменения скорости.
    """
    MUL_COEFF = 0.1

    for pline in polylines:
        for point in pline.points:
            coeff1, coeff2 = MUL_COEFF * abs(point.speed[X]), \
                             MUL_COEFF * abs(point.speed[Y])
            if flag:
                coeff1, coeff2 = -coeff1, -coeff2
            point.speed = (abs(point.speed[X]) + coeff1,
                           abs(point.speed[Y]) + coeff2)


"""Основная программа"""
if __name__ == "__main__":
    pygame.init()
    gameDisplay = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("MyScreenSaver")

    hue = 0
    color = pygame.Color(0)

    runGame = True
    pause = True
    show_help = False

    """данная коллекция была введена для случая нескольких ломаных"""
    lines = []
    polyline = Knot()
    lines.append(polyline)

    while runGame:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runGame = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    runGame = False

                if event.key == pygame.K_r:
                    polyline.points = []

                if event.key == pygame.K_p:
                    pause = not pause

                if event.key == pygame.K_F1:
                    show_help = True

                if event.key == pygame.K_KP_PLUS:
                    polyline.count += 1
                    polyline.get_knot()

                if event.key == pygame.K_KP_MINUS:
                    polyline.count -= 1 if polyline.count > 1 else 0
                    polyline.get_knot()

                if event.key == pygame.K_s:
                    coefficients(polylines=lines)

                if event.key == pygame.K_l:
                    coefficients(polylines=lines, flag=True)

                if event.key == pygame.K_d:
                    if len(polyline.points) > 0:
                        polyline.points.pop()
                        polyline.get_knot()

                if event.key == pygame.K_a:
                    polyline = Knot()
                    lines.append(polyline)

            if event.type == pygame.MOUSEBUTTONDOWN:
                polyline.add_point(current_line=polyline)

        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        color.hsla = (hue, 100, 50, 100)

        for line in lines:
            line.draw_points(line_color=color)

        if not pause:
            for line in lines:
                line.set_points(current_line=line)

        if show_help:
            polyline.draw_help(polylines=lines)

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
