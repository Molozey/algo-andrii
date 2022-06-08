import pygame
import random


# =======================================================================================
# Classes
# =======================================================================================
class Vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v2):
        return Vec2d(self.x + v2.x, self.y + v2.y)

    def __sub__(self, v2):
        return Vec2d(self.x - v2.x, self.y - v2.y)

    def __mul__(self, k):
        if not isinstance(k, Vec2d):
            return Vec2d(self.x * k, self.y * k)
        return self.x * k.x + self.y + k.y

    def __len__(self):
        return (self.x**2 + self.y**2) ** 0.5

    def int_pair(self):
        return int(self.x), int(self.y)


class Polyline:
    def __init__(self, screen_dim=(800, 600)):
        self.points = []
        self.speeds = []
        self.screen_dim = screen_dim

    def set_points(self):
        for p in range(len(self.points)):
            self.points[p] = self.points[p] + self.speeds[p]
            if self.points[p].x > self.screen_dim[0] or self.points[p].x < 0:
                self.speeds[p] = Vec2d(-self.speeds[p].x, self.speeds[p].y)
            if self.points[p].y > self.screen_dim[1] or self.points[p].y < 0:
                self.speeds[p] = Vec2d(self.speeds[p].x, -self.speeds[p].y)

    def draw_points(self, game_display, width=3, color=(255, 255, 255)):
        for p in self.points:
            pygame.draw.circle(game_display, color, p.int_pair(), width)

    def plus_point(self, point):
        self.points.append(point[0])
        self.speeds.append(point[1])

    def minus_point(self):
        if len(self.points) != 0 and len(self.speeds) != 0:
            self.points.pop()
            self.speeds.pop()

    def speed_up(self):
        for i in range(len(self.speeds)):
            self.speeds[i] *= 1.2

    def speed_down(self):
        for i in range(len(self.speeds)):
            self.speeds[i] *= 0.8   # speed > 0


class Knot(Polyline):
    def __init__(self, counter, screen_dim=(800, 600)):
        super().__init__(screen_dim)
        self.counter = counter

    def get_point(self, points, alpha, deg=None):
        if deg is None:
            deg = len(points) - 1
        if deg == 0:
            return points[0]
        return points[deg] * alpha + self.get_point(points, alpha, deg - 1) * (1 - alpha)

    def get_points(self, base_points):
        alpha = 1 / self.counter
        res = []
        for i in range(self.counter):
            res.append(self.get_point(base_points, i * alpha))
        return res

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

    def draw_line(self, game_display, width=3, color=(255, 255, 255)):
        points = self.get_knot()
        for p_n in range(-1, len(points) - 1):
            pygame.draw.line(game_display, color, points[p_n].int_pair(),
                             points[p_n + 1].int_pair(), width)

    def plus_point(self, point):
        super().plus_point(point)
        self.get_knot()

    def minus_point(self):
        super().minus_point()
        self.get_knot()

    def set_points(self):
        super().set_points()
        self.get_knot()

    def plus_counter(self):
        self.counter += 1

    def minus_counter(self):
        if self.counter > 1:
            self.counter -= 1


# =======================================================================================
# Help Window
# =======================================================================================
def draw_help():
    gameDisplay.fill((50, 50, 50))
    font1 = pygame.font.SysFont("courier", 24)
    font2 = pygame.font.SysFont("serif", 24)
    data = []
    data.append(["F1", "Show Help"])
    data.append(["R", "Restart"])
    data.append(["P", "Pause/Play"])
    data.append(["F", "Faster"])
    data.append(["S", "Slower"])
    data.append(["A", "Add new line"])
    data.append(["D", "Delete point (for only the last line)"])
    data.append(["N", "Delete point (for all lines)"])
    data.append(["M", "Add points (one for each line)"])
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
# Functions
# =======================================================================================
def add_line(lines, steps):
    lines.append((Polyline(), Knot(steps)))


# =======================================================================================
# Main
# =======================================================================================
if __name__ == "__main__":
    lines = []
    pygame.init()
    screen_dim = (800, 600)
    gameDisplay = pygame.display.set_mode(screen_dim)
    pygame.display.set_caption("MyScreenSaver")

    steps = 35
    #knot = Knot(steps)
    add_line(lines, steps)

    working = True
    points = []
    speeds = []
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
                    lines = []
                    add_line(lines, steps)
                if event.key == pygame.K_p:
                    pause = not pause
                if event.key == pygame.K_KP_PLUS:
                    lines[-1][1].plus_counter()
                if event.key == pygame.K_F1:
                    show_help = not show_help
                if event.key == pygame.K_KP_MINUS:
                    lines[-1][1].minus_counter()
                if event.key == pygame.K_f:
                    cur_line = lines[-1]
                    cur_line[0].speed_up()
                    cur_line[1].speed_up()
                if event.key == pygame.K_s:
                    cur_line = lines[-1]
                    cur_line[0].speed_down()
                    cur_line[1].speed_down()
                if event.key == pygame.K_a:
                    add_line(lines, steps)
                if event.key == pygame.K_d:
                    lines[-1][0].minus_point()
                    lines[-1][1].minus_point()
                if event.key == pygame.K_n:
                    for line in lines:
                        line[0].minus_point()
                        line[1].minus_point()
                if event.key == pygame.K_m:
                    for line in lines:
                        event_pos = Vec2d(random.randrange(0, 800+1), random.randrange(0, 600))
                        speed = Vec2d(random.random() * 2, random.random() * 2)
                        point_to_add = (event_pos, speed)
                        line[0].plus_point(point_to_add)
                        line[1].plus_point(point_to_add)

            if event.type == pygame.MOUSEBUTTONDOWN:
                event_pos = Vec2d(event.pos[0], event.pos[1])
                speed = Vec2d(random.random() * 2, random.random() * 2)
                point_to_add = (event_pos, speed)
                lines[-1][0].plus_point(point_to_add)
                lines[-1][1].plus_point(point_to_add)

        gameDisplay.fill((0, 0, 0))
        hue = (hue + 1) % 360
        color.hsla = (hue, 100, 50, 100)
        for el in lines:
            el[0].draw_points(game_display=gameDisplay)
            el[1].draw_line(game_display=gameDisplay, color=color)
        if not pause:
            for el in lines:
                el[0].set_points()
                el[1].set_points()
        if show_help:
            draw_help()

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    exit(0)
