class MappingAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def lighten(self, grid):
        dim = (len(grid[0]), len(grid))
        self.adaptee.set_dim(dim)
        __lights, __obstacles = self.analyse_grid(dim, grid)
        self.adaptee.set_lights(__lights)
        self.adaptee.set_obstacles(__obstacles)
        return self.adaptee.grid

    def analyse_grid(self, dim, grid):
        lights, obstacles = list(), list()
        for i in range(dim[0]):
            for j in range(dim[1]):
                if grid[j][i] == 1:
                    lights.append((i, j))
                if grid[j][i] == -1:
                    obstacles.append((i, j))
        return lights, obstacles
