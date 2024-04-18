import chess
import numpy as np
import random
from mpi4py import MPI

def transform_board_to_np(chess_board: chess.Board):
    mapped_pieces = chess_board.piece_map()
    np_board = np.zeros((64,), dtype=np.int32)

    for position, piece in mapped_pieces.items():
        np_board[position] = ord(piece.symbol())
    
    return np_board.reshape((8, 8)) * (1 if chess_board.turn else -1)

def transform_np_to_board(np_board: np.ndarray):
    is_turn = np_board.sum() > 0
    if not is_turn:
        np_board *= -1

    np_board = np_board.flatten()
    mapped_pieces = {idx: chess.Piece.from_symbol(chr(val)) for idx, val in enumerate(np_board) if val != 0}
    
    chess_board = chess.Board()
    chess_board.set_piece_map(mapped_pieces)
    chess_board.turn = is_turn

    return chess_board

def scatter_boards_among_processes(board_list: list):
    # Distribute a list of boards among processes.
   
    # get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # calculate division
    if rank == 0:
        boards = np.ascontiguousarray([transform_board_to_np(board) for board in board_list])
        split = np.array_split(boards, size, axis=0)

        chunk_sizes = np.array([len(s) for s in split]) * 64
        chunk_disps = np.cumsum(chunk_sizes)
        chunk_disps = np.roll(chunk_disps, 1)
        chunk_disps[0] = 0
    else:
        boards = None
        chunk_sizes, chunk_disps = None, None
    
    # broadcast division
    chunk_sizes = comm.bcast(chunk_sizes, root=0)
    chunk_disps = comm.bcast(chunk_disps, root=0)

    # scatter data
    my_boards = np.zeros(chunk_sizes[rank], dtype=np.int32).reshape(-1, 8, 8)
    comm.Scatterv([boards, chunk_sizes, chunk_disps, MPI.INT32_T], my_boards, root=0)

    temp_boards = []
    for board in my_boards:
        temp_boards.append(transform_np_to_board(board))

    return temp_boards

def gather_scores_from_processes(my_scores_list: list, total_num: int):
    # Gather a list of scores among processes.   
   
    # get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        scores = np.ascontiguousarray(np.zeros((total_num, 1), dtype=np.int32))
        split = np.array_split(scores, size, axis=0)

        chunk_sizes = np.array([len(s) for s in split]) * 1
        chunk_disps = np.cumsum(chunk_sizes)
        chunk_disps = np.roll(chunk_disps, 1)
        chunk_disps[0] = 0
    else:
        scores = np.array([], dtype=np.int32)
        chunk_sizes, chunk_disps = None, None
    
    # gather data
    my_scores = np.ascontiguousarray(my_scores_list, dtype=np.int32)
    comm.Gatherv(my_scores, [scores, chunk_sizes, chunk_disps, MPI.INT32_T], root=0)

    scores = scores.flatten().tolist()
    return scores
