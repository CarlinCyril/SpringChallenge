import sys
from enum import Enum
from random import randrange, choice
from typing import Union, Dict, Set, Optional, List


WIDTH = 0
HEIGHT = 0

MOVE = "MOVE"
SPEED = "SPEED"
SWITCH = "SWITCH"


def error(s: str):
    print(s, file=sys.stderr)


class TileType(Enum):
    WALL = '#'
    FLOOR = ' '


class PacShape(Enum):
    ROCK = "ROCK"
    PAPER = "PAPER"
    SCISSORS = "SCISSORS"
    DEAD = "DEAD"


PAYOFF = {PacShape.ROCK: {PacShape.ROCK: 0, PacShape.PAPER: -1, PacShape.SCISSORS: 1},
          PacShape.PAPER: {PacShape.ROCK: 1, PacShape.PAPER: 0, PacShape.SCISSORS: -1},
          PacShape.SCISSORS: {PacShape.ROCK: -1, PacShape.PAPER: 1, PacShape.SCISSORS: 0}}


class Action:
    def __init__(self, pac_id: int) -> None:
        self.pac_id = pac_id
        self.type = None

    def print_action(self) -> str:
        pass


class Position:
    def __init__(self, x_coord: int, y_coord: int) -> None:
        self.x = x_coord
        self.y = y_coord

    def manhattan_distance(self, pos):
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

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


class Grid(dict):
    def __getitem__(self, item):
        if item.x >= WIDTH and super().__getitem__(Position(WIDTH - 1, item.y)).type == TileType.FLOOR:
            error(f"Found a tunnel for {item}")
            return super().__getitem__(Position(item.x % WIDTH, item.y))
        else:
            return super().__getitem__(item)

    def get(self, item, default_value=None):
        if item.x >= WIDTH and super().__getitem__(Position(WIDTH - 1, item.y)).type == TileType.FLOOR:
            error(f"Found a tunnel for {item}")
            return super().get(Position(item.x % WIDTH, item.y), default_value)
        else:
            return super().get(item, default_value)


ADJACENCY = (Position(1, 0), Position(0, -1), Position(-1, 0), Position(0, 1))


class Item:
    def __init__(self, position):
        self.position = position


class Move(Action):
    def __init__(self, pac_id: int, target: Position) -> None:
        super().__init__(pac_id)
        self.type = MOVE
        self.target_position = target

    def print_action(self) -> str:
        return "MOVE {} {} {}".format(self.pac_id, int(self.target_position.x), int(self.target_position.y))


class Switch(Action):
    def __init__(self, pac_id: int, shape: PacShape):
        super().__init__(pac_id)
        self.type = SWITCH
        self.target_shape = shape

    def print_action(self):
        return "SWITCH {} {}".format(self.pac_id, self.target_shape.value)


class Speed(Action):
    def __init__(self, pac_id: int):
        self.type = SPEED
        super().__init__(pac_id)

    def print_action(self):
        return "SPEED {}".format(self.pac_id)


class Tile(Position):
    def __init__(self, x_coord: int, y_coord: int, tile_type: TileType) -> None:
        super().__init__(x_coord, y_coord)
        self.neighbors = set()  # type: Set[Tile]
        self.type = tile_type
        self.value = 0.9
        self._occupant = None  # type: Union[Item, None]
        self.visited = False

    def __eq__(self, other):
        if isinstance(other, Tile):
            return super().__eq__(other) and other.type == self.type
        else:
            return False

    def __repr__(self):
        return f"{super().__repr__()}: {self.value}"

    @property
    def occupant(self):
        return self._occupant

    @occupant.setter
    def occupant(self, occupant: Item):
        if isinstance(occupant, Pacman) or not occupant:
            self.value = 0
        elif isinstance(occupant, Pellet):
            self.value = occupant.value
        else:
            raise ValueError("Unknown occupant type {}".format(type(occupant)))
        self._occupant = occupant

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

    def path_length(self) -> int:
        path_length = 0
        node = self
        while node.parent:
            node = node.parent
            path_length += 1
        return path_length


class Pacman(Item):
    def __init__(self, position: Position, pac_type: PacShape, owned: bool, pac_id: int, speed_turns_left: int,
                 ability_cd: int) -> None:
        super().__init__(position)
        self.id = pac_id
        self.owned = owned
        self.type = pac_type
        self.speed_turns_left = speed_turns_left
        self.ability_cd = ability_cd

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id and self.owned == other.owned

    def __repr__(self):
        return f"Pac {self.id} ({self.position})"

    def attack(self, close_enemy) -> Action:
        if PAYOFF[self.type][close_enemy.type] in (0, -1):
            if self.ability_cd:
                error("I have nothing to do but run away")
                return self.runaway(close_enemy)
            else:
                if close_enemy.ability_cd:
                    error("Lets change shape for battle")
                    counter_shape = next(shape for shape in PacShape if PAYOFF[shape][close_enemy.type] == 1)
                    return Switch(self.id, counter_shape)
                else:
                    error("Preventive run")
                    return self.runaway(close_enemy)
        else:
            if close_enemy.ability_cd:
                error("We can attack")
                return Move(self.id, close_enemy.position)
            else:
                error("Running away just in case")
                return self.runaway(close_enemy)

    def runaway(self, close_enemy) -> Action:
        global WIDTH
        global HEIGHT
        retreat_position = Position(
                (2 * self.position.x - close_enemy.position.x) % WIDTH,
                (2 * self.position.y - close_enemy.position.y) % HEIGHT
            )
        return Move(self.id, retreat_position)


class Pellet(Item):
    def __init__(self, position: Position, value_pellet: int) -> None:
        super().__init__(position)
        self.value = value_pellet

    def __str__(self):
        return "Pellet {} with value {}".format(self.position, self.value)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.value < other.value


class Board:
    def __init__(self) -> None:
        self.width, self.height = [int(i) for i in input().split()]
        global WIDTH
        global HEIGHT
        WIDTH = self.width
        HEIGHT = self.height
        error("W = {}\nH = {}".format(self.width, self.height))
        self.grid = Grid()  # type: Grid[Position, Tile]
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
            for adjacent_position in ADJACENCY:
                sentinel = self.grid.get(position + adjacent_position, Tile(0, 0, TileType.WALL))
                if sentinel.type == TileType.FLOOR:
                    tile.neighbors.add(sentinel)

    def set_occupant(self, position: Position, occupant: Item) -> None:
        self.grid[position].occupant = occupant

    def reset_occupant(self, position: Position) -> None:
        self.grid[position].occupant = None

    def reset_all_occupants(self) -> None:
        for tile in self.grid.values():
            tile.occupant = None

    def distance(self, start: Tile, target: Tile, visited_tiles=None) -> int:
        if visited_tiles is None:
            visited_tiles = set()
        if start == target:
            return 1

        unvisited_tiles = (tile for tile in start.neighbors if tile not in visited_tiles)
        visited_tiles.add(start)
        min_distance = self.width * self.height

        if unvisited_tiles:
            for neighbor in unvisited_tiles:
                distance_neighbor = self.distance(neighbor, target, visited_tiles)
                min_distance = distance_neighbor if distance_neighbor < min_distance else min_distance

        return 1 + min_distance

    # A* search
    def astar_search(self, start: Tile, end: Tile) -> Optional[List[Tile]]:
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
                neighbor.start_distance = neighbor.tile.manhattan_distance(start)
                neighbor.target_distance = neighbor.tile.manhattan_distance(end)
                neighbor.total_cost = neighbor.start_distance + neighbor.target_distance

                # Check if neighbor is in open list and if it has a lower f value
                if neighbor not in visited_nodes:
                    # Everything is green, add neighbor to open list
                    nodes_to_visit.append(neighbor)

        # Return None, no path is found
        return None

    def best_path(self, start: Position, max_distance: int = 1, visited=None) -> Node:
        nodes_to_visit = list()  # type: List[Node]
        nodes_visited = list()  # type: List[Node]

        if visited:
            for visited_tile in visited:
                nodes_visited.append(Node(visited_tile, None))

        start_node = Node(self.grid[start], None)

        nodes_to_visit.append(start_node)

        while nodes_to_visit:
            nodes_to_visit.sort()
            current_node = nodes_to_visit.pop(0)
            nodes_visited.append(current_node)

            error("------------------------------")
            error(f"Current node = {current_node}")

            neighbors = current_node.tile.neighbors
            error(f"Neighbors = {neighbors}")

            for neighbor in neighbors:
                new_node = Node(neighbor, current_node)

                new_node.total_cost = neighbor.value + current_node.total_cost

                if new_node.path_length() <= max_distance and not isinstance(neighbor.occupant, Pacman):
                    if new_node not in nodes_visited:
                        nodes_to_visit.append(new_node)
                    else:
                        error(f"Node {new_node} being discarded")

        potential_nodes = [node for node in nodes_visited if node.path_length() > 0]
        potential_nodes.sort(reverse=True)
        if potential_nodes:
            best_node = potential_nodes[0]
        else:
            best_node = start_node
        return best_node

    def in_sight(self, pac: Pacman, enemy_pac: Pacman) -> bool:
        # error("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # error(f"Check enemy pac {enemy_pac}")
        for unit_move in ADJACENCY:
            current_position = pac.position + unit_move
            current_tile = self.grid.get(current_position, Tile(0, 0, TileType.WALL))
            while current_tile.type == TileType.FLOOR and current_position.x < self.width:
                # error(f"Current tile in sight for pac {pac} = {current_tile}")
                if isinstance(current_tile.occupant, Pacman) and current_tile.occupant == enemy_pac:
                    return True
                current_position += unit_move
                current_tile = self.grid.get(current_position, Tile(0, 0, TileType.WALL))
        return False


class Game:
    def __init__(self):
        self.board = Board()
        self.my_score = 0
        self.opponent_score = 0
        self.my_pacs = set()  # type: Set[Pacman]
        self.known_pellet = dict()  # type: Dict[Position, Pellet]
        self.enemy_pacs = set()  # type: Set[Pacman]
        self.target_moves = dict()  # type: Dict[Pacman, Action]
        self.previous_positions = dict()  # type: Dict[Pacman, Position]
        self.previous_moves = dict()  # type: Dict[Pacman, Action]

    def update(self):
        self.reset()
        self.my_score, self.opponent_score = [int(i) for i in input().split()]
        self.update_pacmen()
        self.update_pellets()

    def update_pacmen(self) -> None:
        visible_pac_count = int(input())  # all your pacs and enemy pacs in sight
        for i in range(visible_pac_count):
            pac_id, mine, x, y, type_id, speed_turns_left, ability_cooldown = input().split()
            pac_id = int(pac_id)
            pac_type = PacShape(type_id)
            mine = mine != "0"
            position = Position(int(x), int(y))
            speed_turns_left = int(speed_turns_left)
            ability_cooldown = int(ability_cooldown)
            new_pac = Pacman(
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

    def update_pellets(self) -> None:
        visible_pellet_count = int(input())  # all pellets in sight
        visible_pellets = self.visible_pellets()
        for i in range(visible_pellet_count):
            x, y, value = [int(j) for j in input().split()]
            position = Position(x, y)
            new_pellet = Pellet(position, value)
            self.known_pellet[position] = new_pellet
            self.board.set_occupant(position, new_pellet)
        # error(f"Pellets being updated {self.known_pellet.keys()}")
        for known_position in visible_pellets.difference(self.known_pellet.keys()):
            # error(f"Deleting pellet at tile {self.board.grid[known_position]}")
            if not isinstance(self.board.grid[known_position].occupant, Pacman):
                self.board.reset_occupant(known_position)

    def next_move(self):
        for pac in self.my_pacs:
            close_enemy = self.enemy_in_sight(pac)
            if close_enemy:
                error(f"Enemy in sight : {close_enemy} for pacman {pac}")
                self.target_moves[pac] = pac.attack(close_enemy)
            elif self.is_stuck(pac):
                error(f"{pac} is stuck")
                position = self.get_random_position(pac)
                self.target_moves[pac] = Move(pac.id, position)
            elif pac.ability_cd == 0:
                self.target_moves[pac] = Speed(pac.id)
            else:
                previous_path = self.board.astar_search(
                    self.board.grid[pac.position], self.board.grid[self.previous_positions[pac]]
                )
                best_node = self.board.best_path(
                    pac.position,
                    (pac.speed_turns_left > 0) + 3,
                    previous_path
                )
                error(f"best node chosen = {best_node}")
                if best_node and best_node.total_cost:
                    node = best_node
                    while node.parent:
                        # error(f"Deleting occupant of node {node}")
                        self.known_pellet.pop(node.tile.get_position(), None)
                        self.board.reset_occupant(node.tile.get_position())
                        node = node.parent
                    self.target_moves[pac] = Move(pac.id, best_node.tile)
                else:
                    best_pellet = self.best_closest_pellet(pac)
                    if best_pellet:
                        error(f"Found a known pellet : {best_pellet}")
                        self.target_moves[pac] = Move(pac.id, best_pellet.position)
                    else:
                        position = self.get_random_position(pac)
                        self.target_moves[pac] = Move(pac.id, position)

    def print_actions(self):
        action_string = " | ".join([action.print_action() for action in self.target_moves.values()])
        print(action_string)

    def reset(self):
        # self.board.reset_all_occupants()
        self.previous_positions = {pac: pac.position for pac in self.my_pacs}
        self.previous_moves = self.target_moves.copy()
        self.my_pacs.clear()
        self.enemy_pacs.clear()
        self.target_moves.clear()
        self.known_pellet.clear()

    def enemy_in_sight(self, pac: Pacman) -> Union[None, Pacman]:
        enemies_in_sight = list()
        for enemy_pac in self.enemy_pacs:
            if self.board.in_sight(pac, enemy_pac):
                enemies_in_sight.append(enemy_pac)
                error(f"Enemies in sight for pacman {pac}: {enemies_in_sight}")
        error(f"Enemies in sight for pacman {pac}: {enemies_in_sight}")
        if enemies_in_sight:
            enemies_in_sight.sort(key=lambda enemy: enemy.position.manhattan_distance(pac.position))
            closest_enemy = enemies_in_sight[0]
            if closest_enemy.position.manhattan_distance(pac.position) < 4:
                return closest_enemy

    def is_stuck(self, pac) -> bool:
        if self.previous_positions:
            # error(f"Previous positions: {self.previous_positions[pac]}")
            # error(f"Previous move: {self.previous_moves[pac].type}")
            return pac.position == self.previous_positions[pac] and self.previous_moves[pac].type == MOVE
        else:
            return False

    def get_random_position(self, pac: Pacman) -> Position:
        random_position = Position(randrange(0, self.board.width), randrange(0, self.board.height))
        random_counter = 0
        while not self._valid_random_position(random_position, pac) or random_counter < 500:
            random_position = Position(randrange(0, self.board.width), randrange(0, self.board.height))
            random_counter += 1
        error(f"No pellet found, sending to a random position: {random_position}")
        return random_position

    def _valid_random_position(self, position: Position, pac: Pacman) -> bool:
        tile = self.board.grid[position]
        valid = tile.type == TileType.FLOOR
        valid &= self.optimal_dispatching(position, pac)
        valid &= tile.value > 0
        return valid

    def optimal_dispatching(self, new_position: Position, pac: Pacman) -> bool:
        width = self.board.width
        height = self.board.height
        my_copy_pacs = self.my_pacs.copy()
        my_copy_pacs.remove(pac)

        avg_position = sum(pac.position for pac in my_copy_pacs)
        avg_position += new_position
        avg_position.x /= len(my_copy_pacs) + 1
        avg_position.y /= len(my_copy_pacs) + 1
        # error(f"Barycenter = {avg_position}")

        is_x_centered = width / 2 - 15 * (width / 100) <= avg_position.x <= width + 15 * (width / 100)
        is_y_centered = height / 2 - 15 * (height / 100) <= avg_position.y <= height + 15 * (height / 100)
        return is_x_centered and is_y_centered

    def visible_pellets(self) -> Set[Position]:
        set_visible_pellets = set()
        for pac in self.my_pacs:
            for unit_move in ADJACENCY:
                current_position = pac.position + unit_move
                current_tile = self.board.grid.get(current_position, Tile(0, 0, TileType.WALL))
                while current_tile.type == TileType.FLOOR and current_position.x < self.board.width:
                    # error(f"Current position = {current_tile}")
                    set_visible_pellets.add(current_position)
                    current_position += unit_move
                    current_tile = self.board.grid.get(current_position, Tile(0, 0, TileType.WALL))
        # error(f"Visible pellets = {set_visible_pellets}")
        return set_visible_pellets

    def best_closest_pellet(self, pac: Pacman) -> Optional[Pellet]:
        if not self.known_pellet:
            return
        optimal_pellets = [pellet for position, pellet in self.known_pellet.items()
                           if self.optimal_dispatching(position, pac)]
        if optimal_pellets:
            optimal_pellets.sort(reverse=True)
            return optimal_pellets[0]
        else:
            list_pellet_positions = list(self.known_pellet.keys())
            list_pellet_positions.sort(
                key=lambda pos: pos.manhattan_distance(pac.position)
            )
            return self.known_pellet[list_pellet_positions[0]]

    def optimal_pathing(self):
        copy_grid = None
        start_tile = next(tile for tile in self.board.grid if tile.type == TileType.FLOOR)  # type: Tile
        crossroad = []
        start_node = Node(start_tile, None)
        neighbors = list(start_tile.neighbors)
        if len(neighbors) > 2:
            crossroad.append(start_node)
        current_node = start_node
        current_tile = neighbors[0]
        while current_tile != start_tile:
            current_node.parent = Node(current_tile, None)
            current_node = current_node.parent
            if len(neighbors) > 2:
                crossroad.append(start_node)


class Gene:
    def __init__(self, tile: Tile, next_gene: () = None):
        self.position = tile.get_position()
        self.neighbors = tile.neighbors
        self.next_gene = next_gene


class Chromosome:
    def __init__(self, genes: List[Gene]):
        self.genes = genes
        self.score = 0

    def swap_genes(self):
        index1, index2 = randrange(0, len(self.genes)), randrange(0, len(self.genes))
        self.genes[index1 - 1].next_gene, self.genes[index2 - 1].next_gene = self.genes[index2], self.genes[index1]
        self.genes[index1], self.genes[index2] = self.genes[index2], self.genes[index1]

    def mutate(self):
        index_gene = randrange(0, len(self.genes))
        selected_gene = self.genes[index_gene]
        mutation = choice(selected_gene.neighbors)
        self.genes[index_gene], self.genes[index_gene - 1].next_gene, mutation.next_gene \
            = mutation, mutation, self.genes[index_gene + 1]

    def gene_creation(self):
        selected_gene = choice(self.genes)  # type: Gene
        following_gene = selected_gene.next_gene  # type: Gene
        new_gene = choice(selected_gene.neighbors.union(following_gene.neighbors))  # type: Gene
        selected_gene.next_gene, new_gene.next_gene = new_gene, following_gene

    def fitness(self):
        pass


class Population:
    def __init__(self, list_chromosomes: List[Chromosome]):
        self.chromosomes = list_chromosomes

    def crossover(self):
        pass

    def run_genetic_algorithm(self):
        pass

    def evolve(self):
        pass


class Bisous:
    pass


game = Game()

while True:
    game.update()
    game.next_move()
    game.print_actions()
