from manim import *
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt


df = pd.read_csv('stocks.csv', index_col='date')


risk_free_rate = 0.03

# percent change per period (weekly)
returns = df.pct_change(1).dropna(how='all')

#annual return of each stock
mean_hist_return = (1 + returns).prod() ** (12 / returns.count()) - 1

#covariance of each stock (annualized)
cov = returns.cov() * 12

def portfo_return(weights):
    '''annual historical retrun of portfolio with specific weight'''
    return np.sum(mean_hist_return * weights)

def portfo_volatility(weights):
    '''annual volatility or standard deviation of portfolio with specific weight'''
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

#crete specific number of random portfolios & calculate their risk and returns
num_random_portfo = 2500
p_return = []
p_risk = []
for p in range(num_random_portfo):
    weights = np.random.random(len(df.columns))
    weights /= np.sum(weights)
    p_return.append(portfo_return(weights))
    p_risk.append(portfo_volatility(weights))

p_return = np.array(p_return)
p_risk = np.array(p_risk)

#back-test and plot using matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(p_risk, p_return, c=(p_return - risk_free_rate) / p_risk,
marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

# target : maximize sharpe ratio
def min_func_sharpe(weights):
    return -(portfo_return(weights) - risk_free_rate) / portfo_volatility(weights)

# constraints and bounds (no short sell) for optimizer
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) -1})
weight_bounds = tuple((0,1) for x in range(len(df.columns)))

# equal weights vector
eweights = np.array(len(df.columns)* [1 / len(df.columns)])

# calculate optimal weights
opt = sco.minimize(min_func_sharpe, eweights, method='SLSQP',
                   bounds=weight_bounds, constraints=constraints)

# max sharpe portfolio weights
opt['x'].round(3)

max_sharpe = (portfo_return(opt['x'].round(3)) - risk_free_rate )/ portfo_volatility(opt['x'].round(3))

# minimum risk portfolio
optm = sco.minimize(portfo_volatility, eweights, method='SLSQP',
                    bounds=weight_bounds,constraints=constraints)

# minimum risk sharpe ratio
min_vol = (portfo_return(optm['x'].round(3)) - risk_free_rate) / portfo_volatility(optm['x'].round(3))

# now we estimate effecient frontier
# trets = target return
# tvols = target volatility

cons = ({'type': 'eq', 'fun': lambda x: portfo_return(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bunds = tuple((0,1) for x in weights)
trets = np.linspace(0.05, 0.36, 50)
tvols = []

for tret in trets:
    result = sco.minimize(portfo_volatility, eweights, method='SLSQP',
                         bounds=bunds, constraints=cons)
    tvols.append(result['fun'])

tvols = np.array(tvols)
ies = []

#now we create the animation for these plots
class EFrontier(Scene):
    def construct(self):
        grid = Axes(x_range=[0, 0.75, 0.25],
            y_range=[-0.25, 0.5, 0.25],
            x_length=12,
            y_length=7,
            axis_config={"font_size": 24},
            tips=False).add_coordinates()

        y_label = grid.get_y_axis_label(Tex("expected return").scale(0.5).rotate(90 * DEGREES),
                                        edge=LEFT, direction=LEFT, buff=0.3)
        x_label = grid.get_x_axis_label(Tex("expected volatility").scale(0.5),
                                        edge=DOWN, direction=DOWN, buff=0.3)
        grid_labels = VGroup(x_label, y_label)

        plot = grid.plot_line_graph(x_values=tvols, y_values=trets, add_vertex_dots=False, line_color=ORANGE)

        min_risk = Star(outer_radius=0.08, color=RED).move_to(
            grid.c2p(portfo_volatility(optm['x']), portfo_return(optm['x']))).scale(0.5)
        min_risk.set_fill(RED)

        max_shrpe = Star(outer_radius=0.08, color=YELLOW).move_to(
            grid.c2p(portfo_volatility(opt['x']), portfo_return(opt['x']))).scale(0.5)
        max_shrpe.set_fill(YELLOW)


        self.play(Create(grid))
        self.play(Write(grid_labels), lag_ratio=0.01, run_time=1.5)
        self.wait(1)
        pos_port = Tex('Possible Portfolios').to_edge(UP)
        self.play(Write(pos_port), func_rate=smooth)
        dot1 = VGroup()
        for points in range(2500):
            if (p_return[points] - risk_free_rate) / p_risk[points] <= 0.05:
                dot = Dot(color=BLUE_C).move_to(grid.c2p(p_risk[points], p_return[points])).scale(0.25)
                dot1.add(dot)
            elif (p_return[points] - risk_free_rate) / p_risk[points] > 0.05 and (p_return[points] - risk_free_rate) / \
                    p_risk[points] <= 0.1:
                dot = Dot(color=BLUE_A).move_to(grid.c2p(p_risk[points], p_return[points])).scale(0.25)
                dot1.add(dot)
            elif (p_return[points] - risk_free_rate) / p_risk[points] > 0.1 and (p_return[points] - risk_free_rate) / \
                    p_risk[points] <= 0.15:
                dot = Dot(color=RED_A).move_to(grid.c2p(p_risk[points], p_return[points])).scale(0.25)
                dot1.add(dot)
            elif (p_return[points] - risk_free_rate) / p_risk[points] > 0.15 and (p_return[points] - risk_free_rate) / \
                    p_risk[points] <= 0.2:
                dot = Dot(color=RED_C).move_to(grid.c2p(p_risk[points], p_return[points])).scale(0.25)
                dot1.add(dot)
            else:
                dot = Dot(color=RED_D).move_to(grid.c2p(p_risk[points], p_return[points])).scale(0.25)
                dot1.add(dot)

        ef = Tex('Efficient Frontier').to_edge(UP)
        mr = Text("Min. Volatility Portfolio\n"
                  "Return: 0.055\n"
                  "Risk: 0.163",
                  color=GREEN_C, font_size=14).move_to(grid.c2p(0.163, 0.055)).shift(UP * 0.6)
        ms = Text("Max. Sharpe Portfolio\n"
                  "Return: 0.292\n"
                  "Risk: 0.428"
                  , color=TEAL_C, font_size=14).move_to(grid.c2p(0.428, 0.292)).shift(DOWN * 0.6)

        self.play(FadeIn(dot1), func_rate=linear)
        self.wait(3)

        self.play(FadeOut(pos_port))
        self.wait(1)

        self.play(Write(ef), func_rate=smooth)
        self.play(Create(plot), func_rate=rush_into, run_time=1.5)
        self.wait()
        self.play(FadeOut(dot1))
        self.play(FadeOut(plot))

        self.play(Create(min_risk))
        self.wait()
        self.play(Write(mr))
        self.play(Create(max_shrpe))
        self.wait()
        self.play(Write(ms))
        self.wait(2)
        self.play(Unwrite(VGroup(ef, mr, ms)))
        self.play(FadeOut(min_risk, max_shrpe))
        self.play(Uncreate(VGroup(grid, grid_labels)))

        wei_min = Text('Min. Volatility Portfolio Weights:\n\n'
                       '            TEVA : 19.7%\n\n'
                       '            IBM  : 20.4%\n\n'
                       '            UAA  : 0%\n\n'
                       '            WMT  : 54.9%\n\n'
                       '            XOM  : 5%\n\n'
                       '            TSLA : 0%'
                      , font_size=24, color=GREEN_A).move_to(UP)

        self.play(Write(wei_min))
        self.wait(4)
        self.play(Unwrite(wei_min))

        wei_max = Text('Max. Sharpe Portfolio Weights:\n\n'
                       '          TEVA : 19.7%\n\n'
                       '          IBM  : 20.4%\n\n'
                       '          UAA  : 0%\n\n'
                       '          WMT  : 54.9%\n\n'
                       '          XOM  : 5%\n\n'
                       '          TSLA : 0%'
                       , font_size=24, color=GREEN_A).move_to(UP)

        self.play(Write(wei_max))
        self.wait(4)
        self.play(Unwrite(wei_max))



