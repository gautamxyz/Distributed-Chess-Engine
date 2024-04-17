import chess
import numpy as np
import random
from mpi4py import MPI

def board_to_numpy(board: chess.Board):
    piece_map = board.piece_map()
    board_array = np.zeros((8, 8), dtype=np.int32)

    for square, piece in piece_map.items():
        board_array[square // 8, square % 8] = ord(piece.symbol())
    
    return board_array * (1 if board.turn else -1)

def numpy_to_board(board_array: np.ndarray):
    turn = board_array.sum() > 0
    if not turn:
        board_array *= -1

    piece_map = {}
    for i in range(8):
        for j in range(8):
            if board_array[i, j] != 0:
                piece_map[i * 8 + j] = chess.Piece.from_symbol(chr(board_array[i, j]))
    
    board = chess.Board()
    board.set_piece_map(piece_map)
    board.turn = turn

    return board

def scatter_boards_among_processes(board_list: list):
    """
    Distribute a list of boards among processes.
    To be used as shown below:

    if rank == 0:
        boards_list = [...]
    else:
        boards_list = None

    my_boards = distribute_among_processes(boards_list)
    """
    # get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # calculate division
    if rank == 0:
        boards = np.ascontiguousarray([board_to_numpy(board) for board in board_list])
        split = np.array_split(boards, size, axis=0)

        chunk_sizes = np.array([len(s) for s in split]) * 64
        chunk_disps = np.insert(np.cumsum(chunk_sizes), 0, 0)[0:-1]
    else:
        boards = None
        chunk_sizes, chunk_disps = None, None
    
    # broadcast division
    chunk_sizes = comm.bcast(chunk_sizes, root=0)
    chunk_disps = comm.bcast(chunk_disps, root=0)

    # scatter data
    my_boards = np.zeros(chunk_sizes[rank], dtype=np.int32).reshape(-1, 8, 8)
    comm.Scatterv([boards, chunk_sizes, chunk_disps, MPI.INT32_T], my_boards, root=0)

    my_boards = [numpy_to_board(board) for board in my_boards]
    return my_boards

def gather_scores_from_processes(my_scores_list: list, total_num: int):
    """
    Gather a list of scores among processes.
    To be used as shown below:

    if rank == 0:
        boards_list = [...] # full board list
        scores_list = [...]
    else:
        boards_list = []
        scores_list = None

    scores_list = gather_scores_from_processes(scores_list, len(board_list))
    """
    # get MPI info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        scores = np.ascontiguousarray(np.zeros((total_num, 1), dtype=np.int32))
        split = np.array_split(scores, size, axis=0)

        chunk_sizes = np.array([len(s) for s in split]) * 1
        chunk_disps = np.insert(np.cumsum(chunk_sizes), 0, 0)[0:-1]
    else:
        scores = np.array([], dtype=np.int32)
        chunk_sizes, chunk_disps = None, None
    
    # gather data
    my_scores = np.ascontiguousarray(my_scores_list, dtype=np.int32)
    comm.Gatherv(my_scores, [scores, chunk_sizes, chunk_disps, MPI.INT32_T], root=0)

    scores = scores.flatten().tolist()
    return scores
