class MappingAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def lighten(self, grid):
        self.adaptee.set_dim((len(grid[0]), len(grid[1])))
        light = [(pos2, pos1) for pos1, e1 in enumerate(grid) for pos2, e2 in enumerate(e1) if e2 == 1]
        obstacles = [(pos2, pos1) for pos1, e1 in enumerate(grid) for pos2, e2 in enumerate(e1) if e2 == -1]
        self.adaptee.set_lights(light)
        self.adaptee.set_obstacles(obstacles)
        return self.adaptee.generate_lights()
