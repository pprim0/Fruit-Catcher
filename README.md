# Fruit Catcher AI

Artificial Intelligence project (AI 2024/2025 – Project 2) that combines a simple **Pygame** game with intelligent agents trained using:

- A **genetic algorithm** to train a **neural network** that controls the basket.
- A **decision tree** to classify falling items (fruit vs. non-fruit) based on tabular attributes.

The goal of the agent is to **catch good fruits** and avoid items that negatively affect the score.

---

## Features

- 2D game built with Pygame:
  - Background and sprites stored inside the `images/` directory
  - Basket movement controlled by an AI agent
  - Falling items (both fruits and “fake” items)
  
- Custom neural network system (`nn.py`):
  - Configurable hidden-layer architecture
  - Automatic computation of total number of weights
  - Forward pass used to determine the agent’s next action

- Genetic Algorithm (`genetic.py`):
  - Creates an initial population of network weight vectors
  - Evaluates fitness by simulating the game
  - Selects elites, performs crossover and mutation
  - Saves the best-performing agent in `best_individual.txt`

- Decision Tree (`dt.py`):
  - Calculates entropy and information gain
  - Builds a simple decision tree based on categorical attributes
  - Predicts whether an item is a fruit (+1) or not (-1)

- Fruit dataset (`items.csv`, `train.csv`, `test.csv`):
  - Columns: `id`, `name`, `color`, `format`, `is_fruit`

---

## Project Structure

```text
fruit-catcher-students/
├── main.py                  # Program entry point
├── game.py                  # Game logic (Pygame)
├── genetic.py               # Genetic algorithm implementation
├── nn.py                    # Neural network implementation
├── dt.py                    # Decision tree implementation
├── best_individual.txt      # Saved best network weights
├── items.csv                # Fruit and non-fruit item list
├── train.csv                # Decision tree training dataset
├── test.csv                 # Decision tree test dataset
├── log_resultados.txt       # (Optional) training logs
├── melhores_por_geracao.txt # (Optional) best fitness per generation
└── images/
    ├── background.jpg
    ├── basket.png
    └── items/
        ├── 1.png
        ├── ...
        └── 15.png
