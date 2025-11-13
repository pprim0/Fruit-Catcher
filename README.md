# 🍎 Fruit Catcher AI

Project developed for the **Artificial Intelligence course (2024/2025)**.  
Created by **Pedro Primo** and **Miguel Ribeiro**  
(GitHub: https://github.com/MiguelR8504)

This project combines a simple 2D game built with **Pygame** and two AI approaches:

- 🧬 A **Genetic Algorithm** that trains a **Neural Network** to control the basket.
- 🌳 A **Decision Tree** that classifies falling items as fruit or non-fruit.

🎯 **Goal:** Catch the good fruits and avoid the harmful items.

---

## 📂 Project Structure

```text
fruit-catcher-students/
├── main.py
├── game.py
├── genetic.py
├── nn.py
├── dt.py
├── best_individual.txt
├── items.csv
├── train.csv
├── test.csv
└── images/
```

---

## ▶️ How to Run

### Install dependencies:
```bash
pip install pygame numpy
```

### Train the AI agent:
```bash
python main.py --train --population 100 --generations 100 --headless
```

### Run the game with the trained agent:
```bash
python main.py --file best_individual.txt
```

---

## ✨ Summary

This project demonstrates:
- Basic **neural network implementation**
- Use of a **genetic algorithm** to evolve weights
- Construction of a simple **decision tree classifier**
- Integration of AI agents in a **Pygame** environment

---

Made with ❤️ by **Pedro Primo** and **Miguel Ribeiro**
