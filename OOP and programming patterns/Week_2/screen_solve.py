#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame
import random
import math
import unittest
import time

SCREEN_DIM = (800, 600)


class TestVec2d(unittest.TestCase):
    SUB_ADD_CASES = ([[1, 1], [0, 0]], [[2, 2], [12, 10]], [(3, 3), [9, 4]], [(4, 4), [4, 4]])

    def test_init(self):
        cases = (([1, 2]), (1, 2), ([3, 4]))
        cases_result = ("x = 1; y = 2", "x = 1; y = 2", 'x = 3; y = 4')
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                if isinstance(obj, tuple):
                    class_test = Vec2d(obj[0], obj[1])
                if isinstance(obj, list):
                    class_test = Vec2d(obj)
                self.assertEqual(str(class_test), cases_result[i])

    def test_sub(self):
        cases = TestVec2d.SUB_ADD_CASES
        cases_result = [Vec2d(1, 1), Vec2d(-10, -8), Vec2d(-6, -1), Vec2d(0, 0)]
        for i in range(len(cases)):
            obj = cases[i]
            with self.subTest(case=i):
                class_test = Vec2d(obj[0]) - Vec2d(obj[1])
                self.assertEqual(str(class_test), str(cases_result[i]))

    def test_add(self):
        cases = TestVec2d.SUB_ADD_CASES
        cases_result = [Vec2d(1, 1), Vec2d(14, 12), Vec2d(12, 7), Vec2d(8, 8)]
        for i in range(len(cases)):
            obj = cases[i]
            with self.subTest(case=i):
                class_test = Vec2d(obj[0]) + Vec2d(obj[1])
                self.assertEqual(str(class_test), str(cases_result[i]))

    def test_len(self):
        cases = ([1, 2], [0, 0], [999, 1])
        cases_result = [math.sqrt(1 ** 2 + 2 ** 2), 0, math.sqrt(999 ** 2 + 1)]
        for i in range(len(cases)):
            obj = cases[i]
            with self.subTest(case=i):
                class_test = Vec2d(obj).len()
                self.assertEqual(float(class_test), float(cases_result[i]))

    def test_mul(self):
        cases = (([12, 2], [2, 12]), ([10, 5], 2), ([1,2], 0))
        cases_result = [48, Vec2d(20, 10), Vec2d(0, 0)]
        for i in range(len(cases)):
            obj = cases[i]
            with self.subTest(case=i):
                if isinstance(obj[1], list):
                    class_test = Vec2d(obj[0]) * Vec2d(obj[1])
                else:
                    class_test = Vec2d(obj[0]) * obj[1]
                self.assertEqual(str(class_test), str(cases_result[i]))

# =======================================================================================
# Функции для работы с векторами
# =======================================================================================


class Vec2d:
    def __init__(self, first_obj, second_obj=None):
        if second_obj == None:
            self.__x = first_obj[0]
            self.__y = first_obj[1]
        else:
            self.__x = first_obj
            self.__y = second_obj

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def __str__(self):
        return f"x = {self.x}; y = {self.y}"

    def __sub__(self, obj):
        """"возвращает разность двух векторов"""
        return Vec2d(self.x - obj.x, self.y - obj.y)

    def __add__(self, obj):
        """возвращает сумму двух векторов"""
        return Vec2d(self.x + obj.x, self.y + obj.y)

    def len(self):
        """возвращает длину вектора"""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __mul__(self, obj):
        """возвращает произведение вектора на число"""
        if isinstance(obj, Vec2d):
            return self.x * obj.x + self.y * obj.y
        else:
            return Vec2d(self.x * obj, self.y * obj)

    def int_pair(self):
        """возвращает пару координат, определяющих вектор (координаты точки конца вектора),
        координаты начальной точки вектора совпадают с началом системы координат (0, 0)"""
        return (int(self.x), int(self.y))

# =======================================================================================
# Функции отрисовки
# =======================================================================================


class Polyline:
    def __init__(self, type_line="norm"):
        """

        :param type_line: Позволяет убрать сглаживание между опорными точками
        """
        self.__points = list()
        self.__speeds = list()
        if str(type_line).lower() not in ("norm", "lines"):
            raise KeyError("Bad type line")
        self.picture_type = str(type_line)
    """
    @property
    def picture_type(self):
        return self.picture_type
    """
    @property
    def speeds(self):
        """

        :return: Speeds at list format.
        isinstance(speeds[:], Vec2d) = True
        """
        return self.__speeds

    @property
    def points(self):
        """

        :return: Points at list format.
        isinstance(points[:], Vec2d) = True
        """
        return self.__points

    def add_point(self, position: Vec2d, speed: Vec2d):
        self.__points.append(position)
        self.__speeds.append(speed)

    def draw_points(self, width=3, color=(255, 255, 255)):
        """функция отрисовки точек на экране"""
        for point in self.points:
            pygame.draw.circle(gameDisplay, color, point.int_pair(), width)

    def set_points(self):
        """функция перерасчета координат опорных точек"""
        for p in range(len(self.points)):
            self.points[p] += self.speeds[p]
            if self.points[p].x > SCREEN_DIM[0] or self.points[p].y < 0:
                self.speeds[p] = Vec2d(- self.speeds[p].x, self.speeds[p].y)
            if self.points[p].y > SCREEN_DIM[1] or self.points[p].y < 0:
                self.speeds[p] = Vec2d(self.speeds[p].x, -self.speeds[p].y)

def draw_help():
    """функция отрисовки экрана справки программы"""
    gameDisplay.fill((50, 50, 50))
    font1 = pygame.font.SysFont("courier", 24)
    font2 = pygame.font.SysFont("serif", 24)
    data = []
    data.append(["F1", "Show Help"])
    data.append(["R", "Restart"])
    data.append(["P", "Pause/Play"])
    data.append(["Num+", "More points"])
    data.append(["Num-", "Less points"])
    data.append(["", ""])
    data.append([str(steps), "Current points"])

    pygame.draw.lines(gameDisplay, (255, 50, 50, 255), True, [
        (0, 0), (800, 0), (800, 600), (0, 600)], 5)
    for i, text in enumerate(data):
        gameDisplay.blit(font1.render(
            text[0], True, (128, 128, 255)), (100, 100 + 30 * i))
        gameDisplay.blit(font2.render(
            text[1], True, (128, 128, 255)), (200, 100 + 30 * i))


# =======================================================================================
# Функции, отвечающие за расчет сглаживания ломаной
# =======================================================================================
class Knot(Polyline):
    def __init__(self, count, type_line="norm"):
        """

        :param count:
        :param type_line: Позволяет убрать сглаживание между опорными точками.
        Принимает значения: ("norm" == Сглаживание, "lines" == Прямые линии)
        """
        super().__init__(type_line=type_line)
        self.__dotes_count = count

    @property
    def dotes_count(self):
        return self.__dotes_count

    def get_point(self, recur_points, alpha: int or float, deg=None):
        if deg is None:
            deg = len(recur_points) - 1
        if deg == 0:
            return recur_points[0]
        return (recur_points[deg] * alpha) + (self.get_point(recur_points, alpha, deg-1) * (1 - alpha))

    def get_points(self, base_points):
        alpha = 1 / self.dotes_count
        res = []
        for i in range(self.dotes_count):
            res.append(self.get_point(base_points, i * alpha))
        return res

    def set_points(self):
        super().set_points()
        self.get_knot()

    def get_knot(self):
        if len(self.points) < 3:
            return []
        res = []
        for i in range(-2, len(self.points) - 2):
            ptn = []
            ptn.append((self.points[i] + self.points[i + 1]) * 0.5)
            ptn.append(self.points[i + 1])
            ptn.append((self.points[i + 1] + self.points[i + 2]) * 0.5)

            res.extend(self.get_points(ptn))
        return res

    def draw_points(self, width=3, color=(255, 255, 255)):
        if self.picture_type == "norm":
            points = self.get_knot()
        if self.picture_type == 'lines':
            points = self.points
        print("points", len(points))
        for p_n in range(-1, len(points) - 1):
            pygame.draw.line(gameDisplay, color, points[p_n].int_pair(), points[p_n + 1].int_pair(), width)


# =======================================================================================
# Основная программа
# =======================================================================================
if __name__ == "__main__":
    pygame.init()
    gameDisplay = pygame.display.set_mode(SCREEN_DIM)
    pygame.display.set_caption("MyScreenSaver")
    SPEED_PARAM = 10   # Регулирует параметр изменения скорости f(time) = sin(time / var)
    SPEED_PARAM_MULTIPLY = 1.5  # Регулирует параметр изменения скорости f(time) = var * sin(time)
    TYPE_LINE = "norm"  # Функционал описан в docstring классов
    steps = 35
    working = True
    polyline = Polyline(type_line=TYPE_LINE)
    knot = Knot(steps, type_line=TYPE_LINE)
    show_help = False
    pause = True

    hue = 0
    color = pygame.Color(0)

    while working:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                working = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    working = False
                if event.key == pygame.K_r:
                    polyline = Polyline(type_line=TYPE_LINE)
                    knot = Knot(steps, type_line=TYPE_LINE)
                if event.key == pygame.K_p:
                    pause = not pause
                if event.key == pygame.K_KP_PLUS:
                    steps += 1
                if event.key == pygame.K_F1:
                    show_help = not show_help
                if event.key == pygame.K_KP_MINUS:
                    steps -= 1 if steps > 1 else 0

            if event.type == pygame.MOUSEBUTTONDOWN:
                polyline.add_point(Vec2d(event.pos),
                                   Vec2d(SPEED_PARAM_MULTIPLY * math.sin(time.perf_counter() / SPEED_PARAM),
                                         SPEED_PARAM_MULTIPLY * math.sin(time.perf_counter() / SPEED_PARAM)))
                knot.add_point(Vec2d(event.pos),
                               Vec2d(SPEED_PARAM_MULTIPLY * math.sin(time.perf_counter() / SPEED_PARAM),
                                     SPEED_PARAM_MULTIPLY * math.sin(time.perf_counter() / SPEED_PARAM)))

        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        color.hsla = (hue, 100, 50, 100)
        polyline.draw_points()
        knot.draw_points(3, color)
        if not pause:
            polyline.set_points()
            knot.set_points()
        if show_help:
            draw_help()

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
