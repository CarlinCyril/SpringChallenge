import sys
import math
from enum import Enum
from typing import List, Union, Dict


def error(s: str):
    print(s, file=sys.stderr)


class TileType(Enum):
    WALL = '#'
    FLOOR = ' '


class Position:
    def __init__(self, x_coord: int, y_coord: int) -> None:
        self.x = x_coord
        self.y = y_coord

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        else:
            return False


class Item:
    pass


class Tile(Position):
    def __init__(self, x_coord: int, y_coord: int, tile_type: TileType) -> None:
        super().__init__(x_coord, y_coord)
        self.type = tile_type
        self.occupant = None  # type: Union[Item, None]

    def __eq__(self, other):
        if isinstance(other, Tile):
            return other == self and other.type == self.type
        else:
            return False


class Pac(Item):
    def __init__(
            self,
            position: Position,
            pac_type: str,
            owned: bool,
            pac_id: int,
            speed_turns_left: int,
            ability_cd: int
    ) -> None:
        self.id = pac_id
        self.position = position
        self.owned = owned
        self.type = pac_type
        self.speed_turns_left = speed_turns_left
        self.ability_cd = ability_cd


class Pellet(Item):
    def __init__(self, position: Position, value_pellet: int) -> None:
        self.position = position
        self.value = value_pellet


class Board:
    def __init__(self) -> None:
        self.width, self.height = [int(i) for i in input().split()]
        self.grid = dict()  # type: Dict[Position, Tile]
        for i in range(height):
            row_input = input()
            for j in range(width):
                if row_input[j] == TileType.FLOOR.value:
                    self.grid[Position(i, j)] = Tile(i, j, TileType.FLOOR)
                elif row_input[j] == TileType.WALL.value:
                    self.grid[Position(i, j)] = Tile(i, j, TileType.WALL)
                else:
                    raise ValueError("TileType {} is unknown".format(row_input[j]))

    def set_occupant(self, position: Position, occupant: Item):
        self.grid[position].occupant = occupant

    def reset_occupant(self, position: Position):
        self.grid[position].occupant = None

    def reset_all_occupants(self):
        for tile in self.grid.values():
            tile.occupant = None


class Game:
    def __init__(self):
        self.board = Board()
        self.my_score = 0
        self.opponent_score = 0

    def update(self):
        self.my_score, self.opponent_score = [int(i) for i in input().split()]
        visible_pac_count = int(input())  # all your pacs and enemy pacs in sight
        for i in range(visible_pac_count):
            pac_id, mine, x, y, type_id, speed_turns_left, ability_cooldown = input().split()
            pac_id = int(pac_id)
            mine = mine != "0"
            position = Position(int(x), int(y))
            speed_turns_left = int(speed_turns_left)
            ability_cooldown = int(ability_cooldown)
            new_pac = Pac(
                position=position,
                pac_type=type_id,
                pac_id=pac_id,
                owned=mine,
                speed_turns_left=speed_turns_left,
                ability_cd=ability_cooldown
            )
            self.board.set_occupant(position, new_pac)
        visible_pellet_count = int(input())  # all pellets in sight
        for i in range(visible_pellet_count):
            # value: amount of points this pellet is worth
            x, y, value = [int(j) for j in input().split()]
            position = Position(int(x), int(y))
            new_pellet = Pellet(position, value)
            self.board.set_occupant(position, new_pellet)


class Bisous:
    pass


# Grab the pellets as fast as you can!

# width: size of the grid
# height: top left corner is (x=0, y=0)
width, height = [int(i) for i in input().split()]
for i in range(height):
    row = input()  # one line of the grid: space " " is floor, pound "#" is wall

# game loop
while True:
    my_score, opponent_score = [int(i) for i in input().split()]
    visible_pac_count = int(input())  # all your pacs and enemy pacs in sight
    for i in range(visible_pac_count):
        # pac_id: pac number (unique within a team)
        # mine: true if this pac is yours
        # x: position in the grid
        # y: position in the grid
        # type_id: unused in wood leagues
        # speed_turns_left: unused in wood leagues
        # ability_cooldown: unused in wood leagues
        pac_id, mine, x, y, type_id, speed_turns_left, ability_cooldown = input().split()
        pac_id = int(pac_id)
        mine = mine != "0"
        x = int(x)
        y = int(y)
        speed_turns_left = int(speed_turns_left)
        ability_cooldown = int(ability_cooldown)
    visible_pellet_count = int(input())  # all pellets in sight
    for i in range(visible_pellet_count):
        # value: amount of points this pellet is worth
        x, y, value = [int(j) for j in input().split()]

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr)

    # MOVE <pacId> <x> <y>
    print("MOVE 0 15 10")

    # This is a test comment
