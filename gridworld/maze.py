import random
import numpy as np

import matplotlib.pylab as plt
import matplotlib.cm as cm
# The array M is going to hold the array information for each cell.
# The first four coordinates tell if walls exist on those sides
# and the fifth indicates if the cell has been visited in the search.
# M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)


class Maze:

    def __init__(self, n_height, n_width):
        self.type = []
        self.n_height = n_height
        self.n_width = n_width
        self.num_rows = (n_height - 3)//2
        self.num_cols = (n_width - 3)//2
        self.M = np.zeros((self.num_rows, self.num_cols, 5), dtype=np.uint8)
        # The array image is going to be the output image to display
        self.image = np.zeros((self.n_height, self.n_width), dtype=np.uint8)

    def create_maze(self, easy=False):
        # Set starting row and column
        # np.random.seed(seed)
        r = 0
        c = 0
        history = [(r, c)]  # The history is the stack of visited locations
        # Trace a path though the cells of the maze and open walls along the path.
        # We do this with a while loop, repeating the loop until there is no history,
        # which would mean we backtracked to the initial start.
        self.image = np.zeros((self.n_height, self.n_width), dtype=np.uint8)
        while history:
            self.M[r, c, 4] = 1  # designate this location as visited
            # check if the adjacent cells are valid for moving to
            check = []
            if c > 0 and self.M[r, c - 1, 4] == 0:
                check.append('L')
            if r > 0 and self.M[r - 1, c, 4] == 0:
                check.append('U')
            if c < self.num_cols - 1 and self.M[r, c + 1, 4] == 0:
                check.append('R')
            if r < self.num_rows - 1 and self.M[r + 1, c, 4] == 0:
                check.append('D')

            if len(check):  # If there is a valid cell to move to.
                # Mark the walls between cells as open if we move
                history.append([r, c])
                move_direction = random.choice(check)
                if move_direction == 'L':
                    self.M[r, c, 0] = 1
                    c = c - 1
                    self.M[r, c, 2] = 1
                if move_direction == 'U':
                    self.M[r, c, 1] = 1
                    r = r - 1
                    self.M[r, c, 3] = 1
                if move_direction == 'R':
                    self.M[r, c, 2] = 1
                    c = c + 1
                    self.M[r, c, 0] = 1
                if move_direction == 'D':
                    self.M[r, c, 3] = 1
                    r = r + 1
                    self.M[r, c, 1] = 1
            else:  # If there are no valid cells to move to.
                # retrace one step back in history if no move is possible
                r, c = history.pop()
        # Open the walls at the start and finish
        # self.M[0, 0, 0] = 1
        # self.M[self.num_rows - 1, self.num_cols - 1, 2] = 1
        # Generate the image for display
        self.image[1:self.n_height-1,self.n_width-3:self.n_width-1] = 255
        self.image[1, 1:self.n_width-1] = 255
        self.image[self.n_height-3:self.n_height-1, 1:self.n_width-1] = 255
        self.image[1:self.n_height-1, 1] = 255
        for row in range(0, self.num_rows):
            for col in range(0, self.num_cols):
                cell_data = self.M[row, col]
                self.image[2 * row + 1 +(1), 2 * col + 1 +(1)] = 255
                if cell_data[0] == 1:
                    self.image[2 * row + 1 +(1), 2 * col +(1)] = 255
                if cell_data[1] == 1:
                    self.image[2 * row +(1), 2 * col + 1 +(1)] = 255
                if cell_data[2] == 1:
                    self.image[2 * row + 1 +(1), 2 * col + 2 +(1)] = 255
                if cell_data[3] == 1:
                    self.image[2 * row + 2 +(1), 2 * col + 1 +(1)] = 255

        # 用"斑秃"让迷宫松散
        if easy:
            n_road = random.randint(min(self.num_rows, self.num_cols),max(self.num_rows, self.num_cols)*4)
            x = np.random.randint(1, high=self.n_width - 2, size=n_road)
            y = np.random.randint(1, self.n_height - 2, n_road)
            self.image[np.ix_(x,y)] = 255



    def get_image(self):
        return self.image

    def set_point(self, x, y):
        """
        设置该点非墙
        """
        self.image[x+1, y+1] = 255

    def set_points(self, points):
        for x, y in points:
            self.set_point(x, y)

    def get_type(self):
        self.type = []
        for row in range(0, self.image.shape[0]):
            for col in range(0, self.image.shape[1]):
                if self.image[row,col] != 255:
                    self.type.append((row,col,1))
        return self.type


if __name__ == "__main__":
    # Display the result
    m = Maze(28, 28) # 5,5,13,
    m.create_maze(True)
    image = m.get_image()
    print(image)
    t = m.get_type()
    print(t)
    plt.imshow(image, cmap=cm.Greys_r, interpolation='none')
    plt.show()
    print(image)
