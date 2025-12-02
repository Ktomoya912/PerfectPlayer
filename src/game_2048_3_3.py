import random
import sys

import numpy


class State:
    def __init__(self, bd: "State" = None, sc: int | None = None):
        self.board: numpy.ndarray = (
            bd if bd is not None else numpy.zeros([9], dtype="int64")
        )
        self.score = sc if sc is not None else 0

    def initGame(self):
        self.board = numpy.zeros([9], dtype="int64")
        self.score = 0
        self.putNewTile()
        self.putNewTile()

    def clone(self):
        return State(self.board.copy(), self.score)

    def print(self, fp=sys.stdout):
        for j in range(3):
            for i in range(3):
                print(f"{self.board[j*3+i]:3d}", end="", file=fp)
            print("", file=fp)
        print(f"score = {self.score}", file=fp)

    def play(self, dir):
        [self.doUp, self.doRight, self.doDown, self.doLeft][dir]()

    def doUp(self):
        _, s1 = moveTiles(self.board[0::3])
        _, s2 = moveTiles(self.board[1::3])
        _, s3 = moveTiles(self.board[2::3])
        self.score += s1 + s2 + s3

    def doRight(self):
        _, s1 = moveTiles(self.board[2::-1])
        _, s2 = moveTiles(self.board[5:2:-1])
        _, s3 = moveTiles(self.board[8:5:-1])
        self.score += s1 + s2 + s3

    def doDown(self):
        _, s1 = moveTiles(self.board[6::-3])
        _, s2 = moveTiles(self.board[7::-3])
        _, s3 = moveTiles(self.board[8::-3])
        self.score += s1 + s2 + s3

    def doLeft(self):
        _, s1 = moveTiles(self.board[0:3])
        _, s2 = moveTiles(self.board[3:6])
        _, s3 = moveTiles(self.board[6:9])
        self.score += s1 + s2 + s3

    def canMoveTo(self, dir: int):
        return [self.canMoveUp, self.canMoveRight, self.canMoveDown, self.canMoveLeft][
            dir
        ]()

    def canMoveUp(self):
        m, _ = moveTiles(self.board[0::3].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[1::3].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[2::3].copy())
        return m

    def canMoveRight(self):
        m, _ = moveTiles(self.board[2::-1].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[5:2:-1].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[8:5:-1].copy())
        return m

    def canMoveDown(self):
        m, _ = moveTiles(self.board[6::-3].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[7::-3].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[8::-3].copy())
        return m

    def canMoveLeft(self):
        m, _ = moveTiles(self.board[0:3].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[3:6].copy())
        if m:
            return True
        m, _ = moveTiles(self.board[6:9].copy())
        return m

    def putNewTile(self):
        emptycells = list()
        for i in range(9):
            # print(self.board[i])
            if self.board[i] == 0:
                emptycells.append(i)
        pos = random.choice(emptycells)
        self.board[pos] = 1 if random.random() < 0.9 else 2

    def isGameOver(self):
        if self.canMoveUp():
            return False
        if self.canMoveRight():
            return False
        if self.canMoveDown():
            return False
        if self.canMoveLeft():
            return False
        return True


def moveTiles(data):
    """
    @param data: numpy slice with 3 elements
    Move the numbers to the left (smaller index)
    """
    if data[0] == 0:
        if data[1] == 0:
            if data[2] == 0:
                # 000
                return False, 0
            else:
                # 001
                data[0] = data[2]
                data[2] = 0
                return True, 0
        else:
            if data[1] == data[2]:
                # 011
                data[0] = data[1] + 1
                data[1] = data[2] = 0
                return True, 2 ** data[0]
            else:
                # 010 or 012
                data[0] = data[1]
                data[1] = data[2]
                data[2] = 0
                return True, 0
    elif data[0] == data[1]:
        # 110 or 112
        data[0] = data[1] + 1
        data[1] = data[2]
        data[2] = 0
        return True, 2 ** data[0]
    else:
        if data[1] == 0:
            if data[0] == data[2]:
                # 101
                data[0] = data[0] + 1
                data[1] = data[2] = 0
                return True, 2 ** data[0]
            elif data[2] != 0:
                # 102
                data[1] = data[2]
                data[2] = 0
                return True, 0
            else:
                # 100
                return False, 0
        else:
            if data[1] == data[2]:
                # 122
                data[1] = data[2] + 1
                data[2] = 0
                return True, 2 ** data[1]
            else:
                return False, 0


def test1():
    import datetime

    print(datetime.datetime.now())
    for i in range(1):
        for i1 in range(4):
            for i2 in range(4):
                for i3 in range(4):
                    data = numpy.array([i1, i2, i3], dtype="int64")
                    print(f"{data}", end="-->")
                    a, b = moveTiles(data)
                    print(f"{a}, {b}, {data}")
    print(datetime.datetime.now())


def test2():
    bd = State()
    bd.initGame()
    for i in range(10000):
        bd.print()
        if bd.isGameOver():
            print("game over")
            break
        print(
            "UP0: "
            + str(bd.canMoveTo(0))
            + " Right1: "
            + str(bd.canMoveTo(1))
            + " Down2: "
            + str(bd.canMoveTo(2))
            + " Left3: "
            + str(bd.canMoveTo(3))
        )
        # while True:
        #     d = int(input())
        #     #d = random.choice([i for i in range(4)])
        #     if bd.canMoveTo(d):
        #         #print(d)
        #         bd.play(d)
        #         break
        # bd.play(int(input()))
        canMoveDirs = [i for i in range(4) if bd.canMoveTo(i)]
        print(f"canMoveDirs = {canMoveDirs}")
        d = random.choice(canMoveDirs)
        print(f'selectedDir = {["up", "right", "down", "left"][d]}')
        bd.play(d)
        bd.putNewTile()


def test3():
    bd = State()
    bd.initGame()
    bd.print()
    print("---")
    bd2 = bd.clone()
    bd2.print()
    print("---")
    bd.board[5] = 3
    bd.print()
    print("---")
    bd2.print()
    print("---")


if __name__ == "__main__":
    # test1()
    test2()
    # test3()
