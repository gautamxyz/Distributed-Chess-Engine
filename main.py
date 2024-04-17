import chess
import time
import random
import argparse
import sys
from mpi4py import MPI

from moves import make_human_move, make_random_move, make_parallel_move

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    board = chess.Board()

    while not board.is_game_over():
        if rank == 0:
            print("Current Number of Moves:", board.fullmove_number)
            print("To Move:", "White" if board.turn else "Black")
            print("Current Board:")
            if args.prettyprint:
                print(board.unicode(invert_color=args.invert))
            else:
                print(board)
        if board.turn:
            # Human makes move.
            if rank == 0:
                if args.simulate:
                    move = make_random_move(board)
                else:
                    move = make_human_move(board)
                board.push(move)
        else:
            # AI makes move.
            if rank == 0:
                start_time = time.time()
            move = make_parallel_move(board, args.depth, args.method)
            if rank == 0:
                end_time = time.time()
                print(f"Time taken to make move: {end_time - start_time:.6f} seconds.", file=sys.stderr)
                board.push(move)
        
        # used to sync the processes
        board = comm.bcast(board, 0)

    if rank == 0:
        if board.is_insufficient_material():
            print("Draw due to insufficient material.")
        elif board.is_fivefold_repetition():
            print("Draw due to five-fold repitition.")
        elif board.is_seventyfive_moves():
            print("Draw due to seventy-five moves.")
        elif board.is_stalemate():
            print("Stalemate.")
        elif board.is_checkmate():
            if board.turn:
                print("Black Won.")
            else:
                print("White Won.")
        
        if args.prettyprint:
            print(board.unicode(invert_color=args.invert, empty_square="."))
        else:
            print(board)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--prettyprint", action="store_true")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--method", type=str, default="minimax")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)

    main(args)