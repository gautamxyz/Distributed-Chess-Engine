# Distributed Chess Engine

This project aims to distribute the load of finding the next optimal move in chess over different computer nodes. The distribution of the load is done following the Message Passing Interfaced (MPI). Hence, the distribution can easily be done over different processes or different computers.

## Running the Program
This project is built using Python 3.10.6.

### Requirements
The following python packages are required:
- python-chess
- argparse
- mpi4py
- numpy



### Running

Please use `mpirun -n N python main.py` to run the engine.
- Please use `--prettyprint` to print the chess pieces in unicode.
- Please use `--invert` to print the chess pieces in inverted colour. You might have to do this depending on your terminal settings.
- Please use `--depth` to specify the depth till which each compute node should search in it's respective algorithm.
- Please use `--simulate` to run a simulation against a random move maker instead of human input.

## Description of Project

Let `N` be the number of nodes involved in the distributed computing. The root node `0` deals with the making of move on the chess board whether it is with a human or it is in a simulation. When this is happening the other nodes are idle. When the AI has to move, it uses minimax with alpha-beta pruning.

The root node finds all the possible moves to be made from the current state of the board. All the new states possible from these moves are generated. Computation of the best possible move from these one-level-down states is now distributed among `N` processes. The different nodes report the best possible move by searching their tree using the chosen method, and the best possible score to the root node. The root node then decides which move is the best to make and makes it.

For the distribution of the games states, the board is converted into an explicit numpy format so that each board state becomes constant in size. This explicit board state is then converted back into a python object for the tree search.