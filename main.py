import sys
import math
from enum import Enum
from random import randrange
from typing import Union, Dict, Set, Optional, List

# Problematic seeds
# seed=-1877397861809193220


def error(s: str):
    print(s, file=sys.stderr)


class TileType(Enum):
    WALL = '#'
    FLOOR = ' '


class PacShape(Enum):
    ROCK = "ROCK"
    PAPER = "PAPER"
    SCISSORS = "SCISSORS"


PAYOFF = {PacShape.ROCK: {PacShape.ROCK: 0, PacShape.PAPER: -1, PacShape.SCISSORS: 1},
          PacShape.PAPER: {PacShape.ROCK: 1, PacShape.PAPER: 0, PacShape.SCISSORS: -1},
          PacShape.SCISSORS: {PacShape.ROCK: -1, PacShape.PAPER: 1, PacShape.SCISSORS: 0}}


class Action:
    def __init__(self, pac_id: int) -> None:
        self.pac_id = pac_id

    def print_action(self) -> str:
        pass


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


class Move(Action):
    def __init__(self, pac_id: int, target: Position) -> None:
        super().__init__(pac_id)
        self.target_position = target

    def print_action(self) -> str:
        return "MOVE {} {} {}".format(self.pac_id, self.target_position.x, self.target_position.y)


class Switch(Action):
    def __init__(self, pac_id: int, shape: PacShape):
        super().__init__(pac_id)
        self.target_shape = shape

    def print_action(self):
        return "SWITCH {} {}".format(self.pac_id, self.target_shape.value)


class Speed(Action):
    def __init__(self, pac_id: int):
        super().__init__(pac_id)

    def print_action(self):
        return "SPEED {}".format(self.pac_id)


class Tile(Position):
    def __init__(self, x_coord: int, y_coord: int, tile_type: TileType) -> None:
        super().__init__(x_coord, y_coord)
        self.neighbors = set()  # type: Set[Tile]
        self.type = tile_type
        self.occupant = None  # type: Union[Item, None]

    def __eq__(self, other):
        if isinstance(other, Tile):
            return super().__eq__(other) and other.type == self.type
        else:
            return False

    def __hash__(self):
        return super().__hash__()

    def get_position(self):
        return Position(self.x, self.y)


class Node:

    # Initialize the class
    def __init__(self, tile: Tile, parent: ()):
        self.tile = tile
        self.parent = parent
        self.start_distance = 0
        self.target_distance = 0
        self.total_cost = 0

    # Compare nodes
    def __eq__(self, other):
        return self.tile == other.tile

    # Sort nodes
    def __lt__(self, other):
        return self.total_cost < other.total_cost

    # Print node
    def __repr__(self):
        return '({0},{1})'.format(self.tile, self.total_cost)


class Pac(Item):
    def __init__(
            self,
            position: Position,
            pac_type: PacShape,
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

    def attack(self, close_enemy) -> Action:
        if PAYOFF[self.type][close_enemy.type] in (0, -1):
            counter_shape = next(shape for shape in PacShape if PAYOFF[self.type][shape] == 1)
            return Switch(self.id, counter_shape)
        else:
            return Move(self.id, close_enemy.position)


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
        self._init_grid()
        self._init_neighbors()

    def _init_grid(self):
        for j in range(self.height):
            row_input = input()
            error(row_input)
            for i in range(self.width):
                self.grid[Position(i, j)] = Tile(i, j, TileType(row_input[i]))

    def _init_neighbors(self):
        for position, tile in self.grid.items():
            for i in (1, -1):
                sentinel = self.grid.get(Position(position.x + i, position.y), Tile(0, 0, TileType.WALL))
                if sentinel.type == TileType.FLOOR:
                    tile.neighbors.add(sentinel)
            for i in (1, -1):
                sentinel = self.grid.get(Position(position.x, position.y + i), Tile(0, 0, TileType.WALL))
                if sentinel.type == TileType.FLOOR:
                    tile.neighbors.add(sentinel)

    def set_occupant(self, position: Position, occupant: Item) -> None:
        self.grid[position].occupant = occupant

    def reset_occupant(self, position: Position) -> None:
        self.grid[position].occupant = None

    def reset_all_occupants(self) -> None:
        for tile in self.grid.values():
            tile.occupant = None

    def closest_pellet(self, position: Position, min_distance: int = 1) -> Pellet:
        min_distance = self.width * self.height
        closest_pellet = None
        for tile in self.grid.values():
            if isinstance(tile.occupant, Pellet):
                # current_distance = position.distance(tile.get_position())
                best_target_node = self.best_path(position, min_distance + 1)
                current_distance = best_target_node.total_cost
                if min_distance <= current_distance < min_distance:
                    min_distance = position.distance(tile.get_position())
                    closest_pellet = tile.occupant
        error("closest pellet is {}".format(closest_pellet))
        if closest_pellet:
            self.reset_occupant(closest_pellet.position)
        return closest_pellet

    def distance(self, start: Tile, target: Tile, visited_tiles=None) -> int:
        if visited_tiles is None:
            visited_tiles = set()
        unvisited_tiles = (tile for tile in start.neighbors if tile not in visited_tiles)
        if unvisited_tiles:
            if target in unvisited_tiles:
                min_distance = 0
            else:
                distances_neighbors = list(map(self.distance, unvisited_tiles))
                min_distance = min(distances_neighbors)
                next_tile = distances_neighbors.index(min_distance)
        else:
            min_distance = self.width * self.height
        return 1 + min_distance

    # A* search
    def astar_search(self, start, end) -> Optional[List[Tile]]:
        # Create lists for open nodes and closed nodes
        nodes_to_visit = []  # type: List[Node]
        visited_nodes = []  # type: List[Node]

        # Create a start node and an goal node
        start_node = Node(start, None)
        goal_node = Node(end, None)

        # Add the start node
        nodes_to_visit.append(start_node)

        # Loop until the open list is empty
        while nodes_to_visit:

            # Sort the open list to get the node with the lowest cost first
            nodes_to_visit.sort()

            # Get the node with the lowest cost
            current_node = nodes_to_visit.pop(0)

            # Add the current node to the closed list
            visited_nodes.append(current_node)

            # Check if we have reached the goal, return the path
            if current_node == goal_node:
                path = []
                while current_node != start_node:
                    path.append(current_node.tile)
                    current_node = current_node.parent
                # Return reversed path
                return path.reverse()

            # Get neighbors
            neighbors = current_node.tile.neighbors

            # Loop neighbors
            for tile_neighbor in neighbors:

                # Check if the node is a wall
                if tile_neighbor.type == TileType.WALL:
                    continue

                # Create a neighbor node
                neighbor = Node(tile_neighbor, current_node)

                # Check if the neighbor is in the closed list
                if neighbor in visited_nodes:
                    continue

                # Generate heuristics (Manhattan distance)
                neighbor.start_distance = abs(neighbor.tile.x - start_node.tile.x) + abs(
                    neighbor.tile.y - start_node.tile.y)
                neighbor.target_distance = abs(neighbor.tile.x - goal_node.tile.x) + abs(
                    neighbor.tile.y - goal_node.tile.y)
                neighbor.total_cost = neighbor.start_distance + neighbor.target_distance

                # Check if neighbor is in open list and if it has a lower f value
                if neighbor not in visited_nodes:
                    # Everything is green, add neighbor to open list
                    nodes_to_visit.append(neighbor)

        # Return None, no path is found
        return None

    def best_path(self, position: Position, max_distance: int = 1, visited=None) -> Node:
        if visited is None:
            visited = set()
        error("-------------------------------")
        error(f"current position is {position}")
        error(f"steps left = {max_distance}")

        current_tile = self.grid[position]
        neighbors = (tile for tile in current_tile.neighbors if tile not in visited)
        visited.add(current_tile)
        best_node = Node(self.grid[position], None)

        if max_distance and neighbors:
            max_score = -1
            for neighbor in neighbors:
                node = self.best_path(neighbor.get_position(), max_distance - 1, visited)
                if max_score < node.total_cost:
                    max_score = node.total_cost
                    best_node = node

        if isinstance(current_tile.occupant, Pellet):
            best_node.total_cost += current_tile.occupant.value
        error(f"best node = {best_node}")
        return best_node

    def get_random_position(self):
        error("No pellet found, sending to a random position.")
        random_position = Position(randrange(0, self.width), randrange(0, self.height))
        while self.grid[random_position].type == TileType.WALL:
            random_position = Position(randrange(0, self.width), randrange(0, self.height))
        return random_position

    def in_sight(self, pac: Pac, enemy_pac: Pac) -> bool:
        if pac.position.x == enemy_pac.position.x:
            return abs(pac.position.y - enemy_pac.position.y) == \
                   self.distance(self.grid[pac.position], self.grid[enemy_pac.position])
        elif pac.position.y == enemy_pac.position.y:
            return abs(pac.position.x - enemy_pac.position.x) == \
                   self.distance(self.grid[pac.position], self.grid[enemy_pac.position])
        else:
            return False


class Game:
    def __init__(self):
        self.board = Board()
        self.my_score = 0
        self.opponent_score = 0
        self.my_pacs = set()  # type: Set[Pac]
        self.enemy_pacs = set()  # type: Set[Pac]
        self.target_moves = dict()  # type: Dict[Pac, Action]
        self.previous_positions = set()  # type: Set[Pac]

    def update(self):
        self.reset()
        self.my_score, self.opponent_score = [int(i) for i in input().split()]
        visible_pac_count = int(input())  # all your pacs and enemy pacs in sight
        for i in range(visible_pac_count):
            pac_id, mine, x, y, type_id, speed_turns_left, ability_cooldown = input().split()
            pac_id = int(pac_id)
            pac_type = PacShape(type_id)
            mine = mine != "0"
            position = Position(int(x), int(y))
            speed_turns_left = int(speed_turns_left)
            ability_cooldown = int(ability_cooldown)
            new_pac = Pac(
                position=position,
                pac_type=pac_type,
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
            close_enemy = self.enemy_in_sight(pac)
            if close_enemy and pac.ability_cd == 0:
                self.target_moves[pac] = pac.attack(close_enemy)
            elif pac.ability_cd == 0:
                self.target_moves[pac] = Speed(pac.id)
            else:
                best_node = self.board.best_path(pac.position, (pac.speed_turns_left > 0) + 2)
                self.target_moves[pac] = Move(pac.id, best_node.tile)
                self.board.reset_occupant(best_node.tile.get_position())

    def print_actions(self):
        action_string = " | ".join([action.print_action() for action in self.target_moves.values()])
        print(action_string)

    def reset(self):
        self.board.reset_all_occupants()
        self.previous_positions = self.my_pacs.copy()
        self.my_pacs.clear()
        self.enemy_pacs.clear()
        self.target_moves.clear()

    def enemy_in_sight(self, pac: Pac) -> Union[None, Pac]:
        for enemy_pac in self.enemy_pacs:
            if self.board.in_sight(pac, enemy_pac):
                return enemy_pac


class Bisous:
    pass


game = Game()

while True:
    game.update()
    game.next_move()
    game.print_actions()
