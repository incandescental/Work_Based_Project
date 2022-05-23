import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import beta
from numpy.random import default_rng
import datetime as dt

pio.renderers.default = 'browser'

### The data preparation steps have not been included here to maintain system security. The process creates a time
### series for each account, seperable into a list of dates and a list of end of day values which can then be used in
### create an object using the class below.

class Spc:

    def __init__(self, ts_y, max_error, date_range, alpha):
        self.ts = ts_y.astype(float).to_list()
        self.max_error = max_error
        self.date_range = date_range
        self.alpha = alpha

    def sliding_window(self):

        ### Function to apply sliding window algorithm
        ### see Keogh, E., Chu, S. , Hart, D. and Pazzani, M. (2001) for implementation

        # Only allow time series with > 28 days worth of data

        if len(self.ts) < 28:

            return None

        else:

            Seg_TS = []
            thetas = []
            anchors = []

            anchor = 0
            i = 2

            # define maximum segment size as quarter of time series

            cl = round(len(self.ts) / 4)

            # Use sliding window method to create linear approximation of series while mse < maximum error
            # or length less than a quarter of time series

            while anchor + i <= len(self.ts):

                while self.create_segment(self.ts[anchor: anchor + i])[2] < self.max_error and i <= cl:
                    i += 1

                seg, coeff, mse = self.create_segment(self.ts[anchor: anchor + i])

                Seg_TS = np.concatenate([Seg_TS, seg])
                thetas.append(coeff)
                anchors.append(anchor)
                anchor += i

            return Seg_TS, thetas, anchors

    @staticmethod
    def create_segment(sub_ts):

        ### Helper function for sliding window algorithm. Returns linear approximation, coefficients and error for
        ### subset of time series

        # Create time steps

        X = list(range(len(sub_ts)))

        # Calculate slope

        xbar = np.mean(X)
        ybar = np.mean(sub_ts)

        sxy = sum([(xi - xbar) * (yi - ybar) for xi, yi in zip(X, sub_ts)])
        sxx = sum([(xi - xbar)** 2 for xi in X])

        b1 = sxy / sxx

        # Calculate intercept

        b0 = ybar - b1*xbar

        # fit equation

        fittedvalues = [b0 + b1 * Xi for Xi in X]

        # Calculate error

        mse = sum([(yi - yhat) ** 2 for yi, yhat in zip(sub_ts, fittedvalues)]) / len(sub_ts)

        return fittedvalues, (b0, b1), mse

    def t2(self, intercepts, slopes):

        ### Function to calculate T2 values for slope and intercept
        ### See Santos-Fernandez, E. (2012) for implementation
        ### Returns T2 values for each intercept/slope pair and upper control limit

        # Create matrix of moving differences for each variable

        xbar_b0 = np.mean(intercepts)
        xbar_b1 = np.mean(slopes)
        m = len(intercepts)

        V = np.column_stack((np.diff(intercepts), np.diff(slopes)))

        S = 1 / 2 * (V.T @ V) / (m - 1)

        # Find inverse

        S_inv = np.linalg.inv(S)

        # Calculate T2 value for each intercept/ slope pair

        t_vals = []

        for b0, b1 in zip(intercepts, slopes):
            x_matrix = np.array([b0 - xbar_b0, b1 - xbar_b1])
            x_matrix = x_matrix[:, np.newaxis]
            x_prime = x_matrix.T
            t2_y = x_prime @ S_inv @ x_matrix
            t_vals.append(t2_y[0][0])

        # Calculate upper control limit

        q = (2 * (m - 1) ** 2) / (3 * m - 4)
        p = 2
        a = p / 2
        b = (q - p - 1) / 2

        # Only allow positive parameter values - negative values indicate too few instances

        if a > 0 and b > 0:

            UCL = (m - 1) ** 2 / m * beta.ppf(1 - self.alpha, a, b)

            return t_vals, UCL

        else:

            return None, None

    def analyse(self):

        ### Function to apply sliding window and T2 methods. Returns a dictionary of results

        yhat = self.sliding_window()

        if yhat is not None:

            linear_approx = yhat[0]
            intercepts = [x[0] for x in yhat[1]]
            slopes = [x[1] for x in yhat[1]]
            anchors = yhat[2]
            tvals, UCL = self.t2(intercepts, slopes)

            res = {'Time_Series': self.ts,
                   'Linear_Approx': linear_approx,
                   'Intercepts': intercepts,
                   'Slopes' : slopes,
                   'T_Values': tvals,
                   'UCL': UCL,
                   'Anchors' : anchors}

            return res

        else:

            return None

    def plot_control_chart(self):

        ### Function to plot a control chart. Returns Plotly figure.

        results = self.analyse()

        if results is not None and results['T_Values'] is not None:

            ooc, change_points = self.out_of_control()
            date_range = self.date_range
            Time_Series = pd.Series(results['Time_Series'])
            Linear_Approx = pd.Series(results['Linear_Approx'])
            T_Values = pd.Series(results['T_Values'])
            UCL = results['UCL']

            fig_y = make_subplots(rows=3, cols=1, subplot_titles=("Time Series",
                                                                  "Linear Approximation",
                                                                  "T^2 Control Chart"))

            fig_y.add_trace(
                go.Scatter(x=date_range,
                           y=Time_Series,
                           showlegend=False),
                row=1, col=1)

            fig_y.add_trace(go.Scatter(x = date_range.iloc[change_points],
                                       y= Time_Series.iloc[change_points],
                                       name = 'Change Detected',
                                       mode = 'markers'),
                row=1, col=1)

            fig_y.add_trace(
                go.Scatter(x=date_range,
                           y=Linear_Approx,
                           showlegend=False),
                row=2, col=1)

            fig_y.add_trace(
                go.Scatter(x=list(range(len(T_Values))),
                           y=T_Values,
                           showlegend=False),
                row=3, col=1)

            fig_y.add_hline(y=UCL, row=3, line_dash="dash", annotation_text="UCL: " + str(round(UCL, 2)))

            return fig_y

        else:

            return None

    def out_of_control(self):

        ### Function to identify change points where T2 value exceeds UCL. Also translates between the location
        ### of the anchor points associated with a segment exhibiting change and the date at which they occured

        results = self.analyse()
        anchors = pd.Series(results['Anchors'])
        Time_Series = results['Time_Series']

        # Find intervention points

        trigger = [i for i,x in enumerate(results['T_Values'] > results['UCL']) if x]

        ooc = len(Time_Series) - anchors.iloc[trigger].max() <= 28
        change_points = anchors.iloc[trigger]

        return ooc, change_points

### Example usage

spc_obj = Spc(ts_y=bal,
              max_error=250,
              date_range=dat,
              alpha = 0.05)

fig = spc_obj.plot_control_chart()

fig.show()