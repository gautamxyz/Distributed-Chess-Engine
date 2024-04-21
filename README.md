# Distributed Chess Engine

The Distributed Chess Engine project aims to efficiently find the optimal move in chess by distributing the computational load across multiple computer nodes. Utilizing the Message Passing Interface (MPI), the workload distribution can seamlessly extend across various processes or even different computers.

## Running the Program
This project is built using Python 3.10.6.

### Packages used
- python-chess
- argparse
- mpi4py
- numpy

### Running and flags

Execute the command mpirun -n N python main.py to launch the engine.

- Add --prettyprint to display chess pieces in Unicode format.
- Include --invert to display chess pieces in inverted colors (may be necessary depending on terminal settings).
- Specify the depth of search for each compute node with --depth.
- Use --simulate to engage in a simulation against a random move maker instead of human input.

## How the task is distributed

In this distributed computing setup, let N represent the number of nodes involved. The root node (0) oversees move generation on the chessboard, whether through human input or simulation. While this occurs, the remaining nodes remain idle. When it's the AI's turn to move, it employs the minimax algorithm with alpha-beta pruning.

The root node identifies all possible moves from the current board state. These moves generate new board states, and the computation of the best move from these states is distributed across N processes. Each node explores its tree using the selected method and reports the best move and its corresponding score to the root node. Finally, the root node selects and executes the best move.

To facilitate state distribution, the board is converted into an explicit numpy format, ensuring uniform size for each board state. Subsequently, this explicit board state is transformed back into a Python object for tree exploration.