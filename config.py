import chess

PIECES_WEIGHTS = {
    chess.PAWN: 10,
    chess.ROOK: 50,
    chess.KNIGHT: 32,
    chess.BISHOP: 33,
    chess.QUEEN: 90,
    chess.KING: 2000
}

RESULT_WEIGHTS = {
    "WIN": 10000,
    "LOSS": -10000,
    "TIE": 0,
}

