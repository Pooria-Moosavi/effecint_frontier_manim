# effecint_frontier_manim

### Introduction

Welcome to the Efficient Frontier Visualization project! This Python script, powered by the Manim library, offers an engaging animated exploration of the Efficient Frontier in modern portfolio theory. The Efficient Frontier represents a set of optimal portfolios that offer the highest expected returns for a defined level of risk.

In this project, we leverage financial data from a CSV file (e.g., 'stocks.csv') to calculate the returns, volatilities, and weights of various portfolios. The animation showcases the relationship between expected returns and volatility for a range of portfolios, visually illustrating the trade-off between risk and reward.

### Key Features

- **Dynamic Portfolio Analysis:** The animation dynamically generates and analyzes random portfolios, showcasing their risk-return profiles. Each portfolio is color-coded based on its Sharpe ratio, providing insights into its risk-adjusted performance.

- **Efficient Frontier Plot:** Watch as the Efficient Frontier emerges, illustrating the optimal portfolios that maximize returns for a given level of risk. The script utilizes the Manim library to create an interactive plot that dynamically updates as the animation progresses.

- **Portfolio Optimization:** Explore the Minimum Volatility Portfolio and the Maximum Sharpe Ratio Portfolio, highlighting key points on the Efficient Frontier. The animation provides detailed information about these portfolios, including their expected returns, risks, and composition.

- **Weights and Composition:** Gain insights into the composition of the optimized portfolios by examining the weights assigned to each asset. The script visually presents the allocation of assets in the Minimum Volatility and Maximum Sharpe Ratio Portfolios.

### Getting Started

Follow the provided usage instructions to clone the repository, install dependencies, download the necessary dataset, and run the Manim animation script. Customize and extend the code to suit your preferences and explore the fascinating world of portfolio optimization.

Whether you are a finance enthusiast, a student, or a developer curious about financial modeling, this project aims to provide an educational and visually appealing introduction to the concepts of the Efficient Frontier and portfolio optimization. Enjoy the journey into the world of risk and return!

### Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Pooria-Moosavi/effecint_frontier_manim.git
   cd effecint_frontier_manim
   ```

2. **Install Dependencies:**
   - Make sure you have Manim installed. Follow the instructions in the [Manim documentation](https://docs.manim.community/en/stable/installation.html) for installation.
     
   - numpy, pandas, scipy and matplotlib libraries are also necessary for this code functionality:
   ```bash
   pip install numpy pandas scipy matplotlib
   ```

3. **Download Data:**
   - Download the dataset (e.g., 'stocks.csv') and place it in the project directory.

4. **Run the Animation:**
   - Open a terminal and navigate to the project directory.
   - Run the Manim animation script using the following command:
     ```bash
     manim -pql efrontier.py EFrontier
     ```

5. **Explore the Animation:**
   - The animation will visualize the Efficient Frontier, Minimum Volatility Portfolio, and Maximum Sharpe Ratio Portfolio.
   - The plot will show different portfolios with varying risk and returns, color-coded based on their Sharpe ratio.
   - The animation includes information about the Minimum Volatility Portfolio and Maximum Sharpe Ratio Portfolio.

6. **Interact with the Animation:**
   - The animation will display key information about the portfolios and their weights.
   - Watch for the Efficient Frontier, Minimum Volatility Portfolio, and Maximum Sharpe Ratio Portfolio.

7. **Additional Information:**
   - Feel free to explore the code and customize it according to your preferences.
   - Check the comments in the code for detailed explanations of the financial concepts and calculations.
