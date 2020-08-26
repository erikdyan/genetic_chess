import argparse
import math
import random

import chess.pgn

CROSSOVER_RATE = 0.75
GENERATIONS = 200
MUTATION_RATE = 0.005
POPULATION_SIZE = 100
CROSSOVER_POPULATION_SIZE = int(POPULATION_SIZE / 2)

SEARCH_DEPTH = 1


def train(white_wins, black_wins):
    # Process and store training data.
    positions = []
    for file in [white_wins, black_wins]:
        with open(file) as file_:
            while game := chess.pgn.read_game(file_):
                board = chess.Board()
                moves = list(game.mainline_moves())

                try:
                    # Choose a random winning position.
                    for i in range(random.randrange(0 if file == white_wins else 1, len(moves) - 1, 2)):
                        board.push(moves[i])
                    positions.append((board, moves[i + 1]))
                except (IndexError, ValueError):  # `moves` is sometimes empty when reading from a PGN file.
                    continue

    # Initialise population.
    population = [Individual() for _ in range(POPULATION_SIZE)]

    # Run generations.
    for i in range(GENERATIONS):
        for individual in population:
            # 1. Mutate.
            if random.random() < MUTATION_RATE:
                gene = random.choice(list(individual.genes))
                if gene == 'PAWN_VALUE' or gene == 'KNIGHT_VALUE' or gene == 'BISHOP_VALUE' or gene == 'ROOK_VALUE' or \
                        gene == 'QUEEN_VALUE':
                    individual.genes[gene] = random.randint(Individual.GENE_MIN, Individual.MATERIAL_MAX)
                else:
                    individual.genes[gene] = random.randint(Individual.GENE_MIN, Individual.GENE_MAX)

            # 2. Assess fitness.
            individual.reset_fitness()

            correct_moves = 0
            engine = Engine(individual.genes, SEARCH_DEPTH)

            for (position, move) in positions:
                if engine.minimax(position, SEARCH_DEPTH, -math.inf, math.inf, position.turn) == move:
                    correct_moves += 1

            individual.fitness = correct_moves * correct_moves

        # 3. Sort population by fitness.
        population.sort(key=lambda x: x.fitness, reverse=True)

        # 4. Crossover.
        for j in range(CROSSOVER_POPULATION_SIZE):
            parent_one, parent_two = population[j], population[j + 1]
            child_one, child_two = population[POPULATION_SIZE - 1 - j], population[POPULATION_SIZE - 2 - j]

            if random.random() < CROSSOVER_RATE:
                for gene in parent_one.genes:
                    if random.random() < 0.5:
                        child_one.genes[gene] = parent_one.genes[gene]
                        child_two.genes[gene] = parent_two.genes[gene]
                    else:
                        child_one.genes[gene] = parent_two.genes[gene]
                        child_two.genes[gene] = parent_one.genes[gene]

        print(f'Generation {i + 1} of {GENERATIONS}. Best individual:\n{population[0].__str__()}')


NUMBER_OF_FILES = NUMBER_OF_RANKS = 8

RANK_1 = 0
RANK_2 = 1
RANK_3 = 2
RANK_4 = 3
RANK_5 = 4
RANK_6 = 5
RANK_7 = 6
RANK_8 = 7


class Engine:
    MIN_DEPTH = 0

    def __init__(self, parameters, max_depth):
        self.parameters = parameters
        self.max_depth = max_depth

    def minimax(self, position, depth, alpha, beta, maximising):
        if depth == Engine.MIN_DEPTH or position.is_game_over():
            return self.__evaluate(position)

        if maximising:
            max_ = -math.inf
            for move in position.legal_moves:
                position.push(move)
                eval_ = self.minimax(position, depth - 1, alpha, beta, False)
                position.pop()

                if eval_ > max_:
                    best = move
                    max_ = eval_

                alpha = max(alpha, eval_)
                if beta <= alpha:
                    break

            return best if depth == self.max_depth else max_

        else:
            min_ = math.inf
            for move in position.legal_moves:
                position.push(move)
                eval_ = self.minimax(position, depth - 1, alpha, beta, True)
                position.pop()

                if eval_ < min_:
                    best = move
                    min_ = eval_

                beta = min(beta, eval_)
                if beta <= alpha:
                    break

            return best if depth == self.max_depth else min_

    def __evaluate(self, position):
        # Return overwhelmingly large bonus if checkmate.
        if position.is_checkmate():
            return 50000 if position.result() == '1-0' else -50000

        eval_ = 0

        eval_ += self.parameters['MOBILITY'] * (self.__mobility(position, chess.WHITE) -
                                                self.__mobility(position, chess.BLACK))

        eval_ += self.parameters['PAWN_VALUE'] * (self.__number_of_pieces(position, chess.PAWN, chess.WHITE) -
                                                  self.__number_of_pieces(position, chess.PAWN, chess.BLACK))

        eval_ += self.parameters['KNIGHT_VALUE'] * (self.__number_of_pieces(position, chess.KNIGHT, chess.WHITE) -
                                                    self.__number_of_pieces(position, chess.KNIGHT, chess.BLACK))

        eval_ += self.parameters['BISHOP_VALUE'] * (self.__number_of_pieces(position, chess.BISHOP, chess.WHITE) -
                                                    self.__number_of_pieces(position, chess.BISHOP, chess.BLACK))

        eval_ += self.parameters['ROOK_VALUE'] * (self.__number_of_pieces(position, chess.ROOK, chess.WHITE) -
                                                  self.__number_of_pieces(position, chess.ROOK, chess.BLACK))

        eval_ += self.parameters['QUEEN_VALUE'] * (self.__number_of_pieces(position, chess.QUEEN, chess.WHITE) -
                                                   self.__number_of_pieces(position, chess.QUEEN, chess.BLACK))

        eval_ += self.parameters['DOUBLED_PAWN_PENALTY'] * (self.__doubled_pawns(position, chess.BLACK) -
                                                            self.__doubled_pawns(position, chess.WHITE))

        eval_ += self.parameters['ISOLATED_PAWN_PENALTY'] * (self.__isolated_pawns(position, chess.BLACK) -
                                                             self.__isolated_pawns(position, chess.WHITE))

        eval_ += self.parameters['PASSED_PAWN'] * (len(self.__passed_pawns(position, chess.WHITE)) -
                                                   len(self.__passed_pawns(position, chess.BLACK)))

        eval_ += self.parameters['PAWN_ADVANCE'] * (self.__pawn_advance(position, chess.WHITE) -
                                                    self.__pawn_advance(position, chess.BLACK))

        eval_ += self.parameters['PAWN_PASSED_ENEMY_KING'] * (self.__pawns_passed_enemy_king(position, chess.WHITE) -
                                                              self.__pawns_passed_enemy_king(position, chess.BLACK))

        eval_ += self.parameters['KNIGHT_OUTPOST'] * (self.__knight_outposts(position, chess.WHITE) -
                                                      self.__knight_outposts(position, chess.BLACK))

        eval_ += self.parameters['BISHOP_PAIR'] * self.__bishop_pair(position, chess.WHITE)
        eval_ -= self.parameters['BISHOP_PAIR'] * self.__bishop_pair(position, chess.BLACK)

        eval_ += self.parameters['ROOK_7TH_RANK'] * (self.__rooks_7th_rank(position, chess.WHITE) -
                                                     self.__rooks_7th_rank(position, chess.BLACK))

        eval_ += self.parameters['ROOK_ATTACK_KING_FILE'] * (self.__rooks_attack_king_file(position, chess.WHITE) -
                                                             self.__rooks_attack_king_file(position, chess.BLACK))

        eval_ += self.parameters['ROOK_ATTACK_KING_ADJACENT_FILE'] * \
                 (self.__rooks_attack_king_adjacent_file(position, chess.WHITE) -
                  self.__rooks_attack_king_adjacent_file(position, chess.BLACK))

        eval_ += self.parameters['ROOK_BEHIND_PASSED_PAWN'] * (self.__rooks_behind_passed_pawn(position, chess.WHITE) -
                                                               self.__rooks_behind_passed_pawn(position, chess.BLACK))

        eval_ += self.parameters['ROOK_CONNECTED'] * (self.__rooks_connected(position, chess.WHITE) -
                                                      self.__rooks_connected(position, chess.BLACK))

        eval_ += self.parameters['ROOK_OPEN_FILE'] * (self.__rooks_open_file(position, chess.WHITE) -
                                                      self.__rooks_open_file(position, chess.BLACK))

        eval_ += self.parameters['ROOK_SEMI_OPEN_FILE'] * (self.__rooks_semi_open_file(position, chess.WHITE) -
                                                           self.__rooks_semi_open_file(position, chess.BLACK))

        eval_ += self.parameters['KING_NO_ENEMY_PAWNS_ADJACENT'] * \
                 self.__king_no_enemy_pawns_adjacent(position, chess.WHITE)
        eval_ -= self.parameters['KING_NO_ENEMY_PAWNS_ADJACENT'] * \
                 self.__king_no_enemy_pawns_adjacent(position, chess.BLACK)

        eval_ += self.parameters['KING_NO_FRIENDLY_PAWNS_ADJACENT_PENALTY'] * \
                 self.__king_no_friendly_pawns_adjacent(position, chess.WHITE)
        eval_ -= self.parameters['KING_NO_FRIENDLY_PAWNS_ADJACENT_PENALTY'] * \
                 self.__king_no_friendly_pawns_adjacent(position, chess.BLACK)

        return eval_

    def __mobility(self, position, colour):
        switched = False
        if not position.turn == colour:
            position.turn = not position.turn
            switched = True

        mobility = position.legal_moves.count()

        if switched:
            position.turn = not position.turn

        return mobility

    def __number_of_pieces(self, position, piece, colour):
        return len(position.pieces(piece, colour))

    def __doubled_pawns(self, position, colour):
        count = 0
        files = self.__square_files(position.pieces(chess.PAWN, colour))

        for file in files:
            if files.count(file) > 1:
                count += 1

        return count

    def __isolated_pawns(self, position, colour):
        count = 0
        files = self.__square_files(position.pieces(chess.PAWN, colour))

        for file in files:
            if (file == 0 and file + 1 not in files) or \
                    (file == 7 and file - 1 not in files) or \
                    (not (file + 1 in files or file - 1 in files)):
                count += 1

        return count

    def __passed_pawns(self, position, colour):
        passed_pawns = []
        opp_pawns = position.pieces(chess.PAWN, not colour)

        for pawn in position.pieces(chess.PAWN, colour):
            passed = True
            file, rank = self.__square_file(pawn), self.__square_rank(pawn)

            for opp_pawn in opp_pawns:
                opp_file, opp_rank = self.__square_file(opp_pawn), self.__square_rank(opp_pawn)
                if (file == opp_file or file == opp_file + 1 or file == opp_file - 1) and \
                   rank < opp_rank if colour == chess.WHITE else rank > opp_rank:
                    passed = False
                    break

            if passed:
                passed_pawns.append(pawn)

        return passed_pawns

    def __pawn_advance(self, position, colour):
        advance = 0
        for rank in self.__square_ranks(position.pieces(chess.PAWN, colour)):
            advance += rank - RANK_2 if colour == chess.WHITE else - (RANK_7 - rank)

        return advance

    def __pawns_passed_enemy_king(self, position, colour):
        count = 0
        king = self.__square_rank(position.king(not colour))

        for pawn in self.__square_ranks(position.pieces(chess.PAWN, colour)):
            if pawn > king if colour == chess.WHITE else pawn < king:
                count += 1

        return count

    def __knight_outposts(self, position, colour):
        count = 0
        pawns = position.pieces(chess.PAWN, colour)

        for knight in position.pieces(chess.KNIGHT, colour):
            if not set(position.attackers(colour, knight)).isdisjoint(pawns):
                count += 1

        return count

    def __bishop_pair(self, position, colour):
        bishops = position.pieces(chess.BISHOP, colour)
        if len(bishops) < 2:
            return False

        white = black = False
        for bishop in bishops:
            if self.__square_file(bishop) % 2 == 0:
                if self.__square_rank(bishop) % 2 == 0:
                    black = True
                else:
                    white = True

            else:
                if self.__square_rank(bishop) % 2 == 0:
                    white = True
                else:
                    black = True

        return white and black

    def __rooks_7th_rank(self, position, colour):
        ranks = self.__square_ranks(position.pieces(chess.ROOK, colour))
        return ranks.count(RANK_7) if colour == chess.WHITE else ranks.count(RANK_2)

    def __rooks_attack_king_file(self, position, colour):
        count = 0
        king = self.__square_file(position.king(not colour))

        for rook in self.__square_files(position.pieces(chess.ROOK, colour)):
            if rook == king:
                count += 1

        return count

    def __rooks_attack_king_adjacent_file(self, position, colour):
        count = 0
        king = self.__square_file(position.king(not colour))

        for rook in self.__square_files(position.pieces(chess.ROOK, colour)):
            if rook == king + 1 or rook == king - 1:
                count += 1

        return count

    def __rooks_behind_passed_pawn(self, position, colour):
        count = 0
        pawns = self.__square_files(self.__passed_pawns(position, colour))

        for rook in self.__square_files(position.pieces(chess.ROOK, colour)):
            if rook in pawns:
                count += 1

        return count

    def __rooks_connected(self, position, colour):
        count = 0
        rooks = position.pieces(chess.ROOK, colour)

        if len(rooks) < 2:
            return 0

        for rook in rooks:
            if not set(position.attackers(colour, rook)).isdisjoint(rooks):
                count += 1

        return count

    def __rooks_open_file(self, position, colour):
        count = 0
        pawns = self.__square_files(position.pieces(chess.PAWN, colour)) + \
                self.__square_files(position.pieces(chess.PAWN, not colour))

        for rook in self.__square_files(position.pieces(chess.ROOK, colour)):
            if rook not in pawns:
                count += 1

        return count

    def __rooks_semi_open_file(self, position, colour):
        count = 0
        white = self.__square_files(position.pieces(chess.PAWN, chess.WHITE))
        black = self.__square_files(position.pieces(chess.PAWN, chess.BLACK))

        for rook in self.__square_files(position.pieces(chess.ROOK, colour)):
            if (rook in white and rook not in black) or (rook in black and rook not in white):
                count += 1

        return count

    def __king_no_enemy_pawns_adjacent(self, position, colour):
        return bool(self.__adjacent_squares(position.king(colour)).isdisjoint(position.pieces(chess.PAWN, not colour)))

    def __king_no_friendly_pawns_adjacent(self, position, colour):
        return bool(self.__adjacent_squares(position.king(colour)).isdisjoint(position.pieces(chess.PAWN, colour)))

    def __adjacent_squares(self, square):
        adjacent = {
            square + NUMBER_OF_FILES - 1, square + NUMBER_OF_FILES, square + NUMBER_OF_FILES + 1,
            square - 1, square + 1,
            square - NUMBER_OF_FILES - 1, square - NUMBER_OF_FILES, square - NUMBER_OF_FILES + 1
        }

        file = self.__square_file(square)
        if file == 0:
            adjacent -= {square + NUMBER_OF_FILES - 1, square - 1, square - NUMBER_OF_FILES - 1}
        elif file == 7:
            adjacent -= {square + NUMBER_OF_FILES + 1, square + 1, square - NUMBER_OF_FILES + 1}

        rank = self.__square_rank(square)
        if rank == 0:
            adjacent -= {square - NUMBER_OF_FILES - 1, square - NUMBER_OF_FILES, square - NUMBER_OF_FILES + 1}
        elif rank == 7:
            adjacent -= {square + NUMBER_OF_FILES - 1, square + NUMBER_OF_FILES, square + NUMBER_OF_FILES + 1}

        return adjacent

    def __square_files(self, squares):
        return [self.__square_file(square) for square in squares]

    def __square_ranks(self, squares):
        return [self.__square_rank(square) for square in squares]

    def __square_file(self, square):
        return square % NUMBER_OF_FILES

    def __square_rank(self, square):
        return square // NUMBER_OF_RANKS


class Individual:
    GENE_MIN = 1
    GENE_MAX = 50
    MATERIAL_MAX = 1000

    PAWN_VALUE = 100

    def __init__(self):
        self.fitness = 0
        self.genes = {
            'MOBILITY': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'PAWN_VALUE': Individual.PAWN_VALUE,
            'KNIGHT_VALUE': random.randint(Individual.GENE_MIN, Individual.MATERIAL_MAX),
            'BISHOP_VALUE': random.randint(Individual.GENE_MIN, Individual.MATERIAL_MAX),
            'ROOK_VALUE': random.randint(Individual.GENE_MIN, Individual.MATERIAL_MAX),
            'QUEEN_VALUE': random.randint(Individual.GENE_MIN, Individual.MATERIAL_MAX),
            'DOUBLED_PAWN_PENALTY': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ISOLATED_PAWN_PENALTY': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'PASSED_PAWN': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'PAWN_ADVANCE': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'PAWN_PASSED_ENEMY_KING': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'KNIGHT_OUTPOST': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'BISHOP_PAIR': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_7TH_RANK': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_ATTACK_KING_FILE': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_ATTACK_KING_ADJACENT_FILE': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_BEHIND_PASSED_PAWN': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_CONNECTED': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_OPEN_FILE': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'ROOK_SEMI_OPEN_FILE': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'KING_NO_ENEMY_PAWNS_ADJACENT': random.randint(Individual.GENE_MIN, Individual.GENE_MAX),
            'KING_NO_FRIENDLY_PAWNS_ADJACENT_PENALTY': random.randint(Individual.GENE_MIN, Individual.GENE_MAX)
        }

    def __str__(self):
        str_ = f'FITNESS: {self.fitness}\n'
        for gene in self.genes:
            str_ += f'{gene}: {self.genes[gene]}\n'

        return str_

    def reset_fitness(self):
        self.fitness = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('white_wins', help='a PGN file containing games won by white')
    parser.add_argument('black_wins', help='a PGN file containing games won by black')

    args = parser.parse_args()
    train(args.white_wins, args.black_wins)
