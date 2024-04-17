import chess
import random
from math import inf
from mpi4py import MPI
from typing import List

from config import *
from interface import *

def get_score(board: chess.Board, player: bool):
    if board.is_stalemate():# or board.is_fivefold_repetition or board.is_insufficient_material() or board.is_seventyfive_moves():
        return RESULT_WEIGHTS["TIE"]
    elif board.is_checkmate() and board.turn == (not player):
        return RESULT_WEIGHTS["WIN"]
    elif board.is_checkmate() and board.turn == player:
        return RESULT_WEIGHTS["LOSS"]
    else:
        total = random.randint(0, 10)
        for piece, weight in PIECES_WEIGHTS.items():
            pos_cnt = str(board.pieces(piece, player)).count('1')
            neg_cnt = str(board.pieces(piece, not player)).count('1')

            total += (pos_cnt - neg_cnt) * weight

        return total

def sorted_moves(board: chess.Board) -> List[str]:
    NAME_TO_SQUARE = dict(zip(chess.SQUARE_NAMES, chess.SQUARES))

    def square_name(move):
        return move.uci()[:2]

    moves = list(board.legal_moves)

    squares = [NAME_TO_SQUARE[name] for name in map(square_name, moves)]
    pieces = [board.piece_type_at(square) for square in squares]

    moves = sorted(zip(moves, pieces), key=lambda x: x[1], reverse=True)

    return moves

def minimax(board: chess.Board, depth: int=3, alpha: float=-inf, beta: float=+inf):
    player = board.turn
    if depth == 0 or board.is_game_over():
        return get_score(board, player), None
    
    if player:
        # maximizing player
        max_score, best_move = -inf, None
        legal_moves = sorted_moves(board)
        for move, _ in legal_moves:
            new_board = board.copy()
            new_board.push(move)
            score, _ = minimax(new_board, depth - 1, alpha, beta)

            alpha = max(alpha, score)
            if beta <= alpha:
                break

            if score > max_score:
                max_score, best_move = score, move
            elif score == max_score:
                if random.randint(0, 1):
                    max_score, best_move = score, move
        return max_score, best_move
    else:
        # minimizing player
        min_score, best_move = +inf, None
        legal_moves = sorted_moves(board)
        for move, _ in legal_moves:
            new_board = board.copy()
            new_board.push(move)
            score, _ = minimax(new_board, depth - 1, alpha, beta)

            beta = min(beta, score)
            if beta <= alpha:
                break

            if score < min_score:
                min_score, best_move = score, move
            elif score == min_score:
                if random.randint(0, 1):
                    min_score, best_move = score, move
        return min_score, best_move

def make_random_move(board: chess.Board):
    legal_moves = list(board.legal_moves)
    random_move = random.choice(legal_moves)
    return random_move

def make_parallel_move(board: chess.Board, depth: int=3, method: str="minimax"):
    # get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # get all the possibe next game states
        legal_moves = list(board.legal_moves)
        boards_list = []
        for move in legal_moves:
            tmp_board = board.copy()
            tmp_board.push(move)
            boards_list.append(tmp_board)
    else:
        boards_list = []
    
    my_boards = scatter_boards_among_processes(boards_list)

    my_moves = []
    my_scores = []

    for child_board in my_boards:
        if method == "minimax":
            score, move = minimax(child_board, depth)
        else:
            raise NotImplementedError
        my_moves.append(move)
        my_scores.append(score)

    moves_list = gather_moves_from_processes(my_moves, len(boards_list))
    scores_list = gather_scores_from_processes(my_scores, len(boards_list))

    # finally which move should i make?  i am board.turn. i am looking at the
    # scores of all the states one level down. if i am white, i should maximize.
    # if i am black i should minimze.
    if rank == 0:
        best_score = -inf if board.turn else +inf
        best_move = None

        for move, score in zip(legal_moves, scores_list):
            if board.turn:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    else:
        return None
    

def make_human_move(board: chess.Board):
    human_move_str = input("What move do you want to make? ")
    while chess.Move.from_uci(human_move_str) not in board.legal_moves:
        human_move_str = input("Illegal Move! What move do you want to make? ")
    return chess.Move.from_uci(human_move_str)