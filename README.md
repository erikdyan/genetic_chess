# Genetic Chess

A genetic algorithm for evolving a chess evaluation function.

## Evaluation Function

The evaluation function of this program consists of 22 parameters, which covers the key aspects, such as material, mobility, centricity and king safety, of a chess position.

The value of a pawn is fixed at 100, which serves as a reference for all other parameter values. The other four material parameters are initialised to a random value between 1 and 1000, while the remaining parameters are initialised to a random value between 1 and 50.

## Evolution

This program uses a database of 5000 games played between grandmasters rated above 2600 Elo. One winning position (positions where the side to move went on to win) is picked randomly from each game to be used for training.

In each generation, every individual performs a 1-ply search using the evaluation function and its parameter values. The best moved returned by the search is compared to the move played in the actual game. The move is "correct" if it is the same as the move played by the grandmaster, otherwise it is "incorrect". The fitness of the individual is calculated as the square of the total number of correct moves returned across the 5000 positions.

The algorithm uses the following parameters:

* crossover rate = 0.75
* mutation rate = 0.005
* number of generations = 200
* population size = 100
* crossover population size = 50
