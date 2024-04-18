import chess
import time
import random
import argparse
import sys
from mpi4py import MPI

from moves import make_human_move, make_random_move, make_parallel_move

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--prettyprint", action="store_true")
arg_parser.add_argument("--invert", action="store_true")
arg_parser.add_argument("--simulate", action="store_true")
arg_parser.add_argument("--depth", type=int, default=3)
arguments = arg_parser.parse_args()

comm_channel = MPI.COMM_WORLD
process_rank = comm_channel.Get_rank()

chess_board = chess.Board()

while not chess_board.is_game_over():
    if process_rank == 0:
        print("Current Move Number: ", chess_board.fullmove_number)
        print("To Move:", "White" if chess_board.turn else "Black")
        print("Current Board:")
        if arguments.prettyprint:
            print(chess_board.unicode(invert_color=arguments.invert,empty_square="."))
        else:
            print(chess_board)
    if chess_board.turn:
        # Human makes move.
        if process_rank == 0:
            if arguments.simulate:
                selected_move = make_random_move(chess_board)
            else:
                selected_move = make_human_move(chess_board)
            chess_board.push(selected_move)
    else:
        # AI makes move.
        if process_rank == 0:
            start_time = time.time()
        selected_move = make_parallel_move(chess_board, arguments.depth)
        if process_rank == 0:
            end_time = time.time()
            print(f"Time taken to make move: {end_time - start_time:.6f} seconds.", file=sys.stderr)
            chess_board.push(selected_move)
    
    # used to sync the processes
    chess_board = comm_channel.bcast(chess_board, 0)

if process_rank == 0:
    if chess_board.is_insufficient_material():
        print("Draw due to insufficient material.")
    elif chess_board.is_fivefold_repetition():
        print("Draw due to five-fold repitition.")
    elif chess_board.is_seventyfive_moves():
        print("Draw due to seventy-five moves.")
    elif chess_board.is_stalemate():
        print("Stalemate.")
    elif chess_board.is_checkmate():
        if chess_board.turn:
            print("Black Won.")
        else:
            print("White Won.")
    
    if arguments.prettyprint:
        print(chess_board.unicode(invert_color=arguments.invert, empty_square="."))
    else:
        print(chess_board)