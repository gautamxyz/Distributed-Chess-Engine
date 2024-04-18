import chess

PIECES_WEIGHTS = {
    chess.PAWN: 10,
    chess.BISHOP: 30,
    chess.KING: 0,
    chess.KNIGHT: 30,
    chess.ROOK: 50,
    chess.QUEEN: 90,
}

RESULT_WEIGHTS = {
    "WIN": 10000,
    "LOSS": -10000,
    "TIE": 0,
}

STOCKFISH_PATH = "/usr/bin/stockfish"