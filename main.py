import sys
import math
from enum import Enum
from typing import Union, Dict, Set


def error(s: str):
    print(s, file=sys.stderr)


class TileType(Enum):
    WALL = '#'
    FLOOR = ' '


class Position:
    def __init__(self, x_coord: int, y_coord: int) -> None:
        self.x = x_coord
        self.y = y_coord

    def distance(self, pos):
        # Manhattan distance
        return abs(self.x - pos.x) + abs(self.y - pos.y)

    def get_position(self):
        return self

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        else:
            return False

    def __str__(self):
        return f"{self.x} {self.y}"

    def __repr__(self):
        return f"{self.x} {self.y}"

    def __hash__(self):
        return hash((self.x, self.y))


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

    def __hash__(self):
        return self.id.__hash__()


class Pellet(Item):
    def __init__(self, position: Position, value_pellet: int) -> None:
        self.position = position
        self.value = value_pellet

    def __str__(self):
        return "Pellet {} with value {}".format(self.position, self.value)


class Board:
    def __init__(self) -> None:
        self.width, self.height = [int(i) for i in input().split()]
        error("W = {}\nH = {}".format(self.width, self.height))
        self.grid = dict()  # type: Dict[Position, Tile]
        for j in range(self.height):
            row_input = input()
            error(row_input)
            for i in range(self.width):
                if row_input[j] == TileType.FLOOR.value:
                    self.grid[Position(i, j)] = Tile(i, j, TileType.FLOOR)
                elif row_input[j] == TileType.WALL.value:
                    self.grid[Position(i, j)] = Tile(i, j, TileType.WALL)
                else:
                    raise ValueError("TileType {} is unknown".format(row_input[j]))

    def set_occupant(self, position: Position, occupant: Item) -> None:
        self.grid[position].occupant = occupant

    def reset_occupant(self, position: Position) -> None:
        self.grid[position].occupant = None

    def reset_all_occupants(self) -> None:
        for tile in self.grid.values():
            tile.occupant = None

    def closest_pellet(self, position: Position) -> Pellet:
        # Absolute primitive shortest path
        min_distance = self.width * self.height
        closest_pellet = None
        for tile in self.grid.values():
            if isinstance(tile.occupant, Pellet):
                current_distance = position.distance(tile.get_position())
                if current_distance < min_distance:
                    min_distance = position.distance(tile.get_position())
                    closest_pellet = tile.occupant
        error("closest pellet is {}".format(closest_pellet))
        return closest_pellet


class Game:
    def __init__(self):
        self.board = Board()
        self.my_score = 0
        self.opponent_score = 0
        self.my_pacs = set()  # type: Set[Pac]
        self.enemy_pacs = set()  # type: Set[Pac]
        self.target_moves = dict()  # type: Dict[Pac, Position]

    def update(self):
        self.board.reset_all_occupants()
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
            if mine:
                self.my_pacs.add(new_pac)
            else:
                self.enemy_pacs.add(new_pac)
        visible_pellet_count = int(input())  # all pellets in sight
        for i in range(visible_pellet_count):
            x, y, value = [int(j) for j in input().split()]
            position = Position(x, y)
            new_pellet = Pellet(position, value)
            self.board.set_occupant(position, new_pellet)

    def next_move(self):
        for pac in self.my_pacs:
            closest_pellet = self.board.closest_pellet(pac.position)
            self.target_moves[pac] = closest_pellet.position \
                if closest_pellet.position not in self.target_moves.values() else pac.position

    def move(self):
        for pac, target in self.target_moves.items():
            print("MOVE {} {} {}".format(pac.id, target.x, target.y))


class Bisous:
    pass


game = Game()

while True:
    game.update()
    game.next_move()
    game.move()
