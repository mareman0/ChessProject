import pandas as pd
import chess

def format_moves(i):
    moves = list(debutes['moves'][i].split())
    del moves[::3]
    moves = list(map(lambda x: x[:2] + x[3:], moves))
    for move in range(len(moves)):
        if moves[move] == '0-0':
            if move % 2 == 0:
                moves[move] = 'e1g1'
            else:
                moves[move] = 'e8g8'
        elif moves[move] == '0-0-':
            if move % 2 == 0:
                moves[move] = 'e1c1'
            else:
                moves[move] = 'e8c8'
    return moves


debutes = pd.read_csv('debutes_database.csv', usecols=['eco_num', 'name', 'moves'])

print(debutes)

debutes_FENs = pd.Series(name='FEN', dtype=str)
debutes = debutes.append(debutes_FENs.to_frame())
for debute in range(2006):
    starting_board = chess.Board()
    for i in format_moves(debute):
        m = chess.Move.from_uci(i)
        starting_board.push(m)
    debutes['FEN'][debute]=starting_board.fen()
debutes.to_csv('debutes_database.csv', index=False)
