
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class LogitSVMMargin(object):

    def __init__(self, fname, xnames, yname):
        self.df = pd.read_csv(fname)
        self.X = self.df[xnames].values
        self.y = self.df[yname].values
        self.xnames = xnames
        self.yname = yname


    def plot_points(self, size=None, show=False):
        if size == None:
            plt.scatter(self.df[self.xnames[0]], self.df[self.xnames[1]], c=self.df[self.yname])
        else:
            plt.scatter(self.df[self.xnames[0]], self.df[self.xnames[1]], c=self.df[self.yname], s=size)
        plt.xlabel(self.xnames[0])
        plt.ylabel(self.xnames[1])
        if show:
            plt.show()

    def fit_logit(self):
        logistic = LogisticRegression()
        logistic.fit(self.X, self.y)
        return logistic

    def fit_svc(self):
        svc = SVC(kernel='linear')
        svc.fit(self.X, self.y)
        return svc

    def plot_logit_decision(self, logit, show=True):
        w = logit.coef_[0]
        a = -w[0] / w[1]
        x = self.df[self.xnames[0]]
        y = self.df[self.xnames[1]]
        x_min = min(x) - np.std(x)
        x_max = max(x) + np.std(x)
        y_min = min(y) - np.std(y)
        y_max = max(y) + np.std(y)
        xx = np.linspace(x_min, x_max)
        yy = a * xx - (logit.intercept_[0]) / w[1]
        plt.plot(xx, yy, 'k-')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        if show:
            plt.show()

    def plot_svc_decision(self, svc, show=True):
        # get the separating hyperplane
        w = svc.coef_[0]
        a = -w[0] / w[1]
        x = self.df[self.xnames[0]]
        y = self.df[self.xnames[1]]
        x_min = min(x) - np.std(x)
        x_max = max(x) + np.std(x)
        y_min = min(y) - np.std(y)
        y_max = max(y) + np.std(y)
        xx = np.linspace(x_min, x_max)
        yy = a * xx - (svc.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors
        margin = 1 / np.sqrt(np.sum(svc.coef_ ** 2))
        yy_down = yy + a * margin
        yy_up = yy - a * margin

        # plot the line, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-')
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        if show:
            plt.show()

    def calc_margin(self, est):
        w = est.coef_
        intercept = est.intercept_
        distances = np.array([np.linalg.norm(intercept + x.dot(w.T)) / np.linalg.norm(w) for x in self.X])
        print np.array([np.linalg.norm(intercept + x.dot(w.T)) for x in self.X])
        return distances


if __name__ == '__main__':
    obj = LogitSVMMargin('../data/data_scientist.csv', ['gym_hours', 'email_hours'], 'data_scientist')
    obj.plot_points(size=None, show=True)

    logit = obj.fit_logit()
    logit_margin = obj.calc_margin(logit)
    obj.plot_points(size=logit_margin, show=False)
    obj.plot_logit_decision(logit, show=True)

    svc = obj.fit_svc()
    svc_margin = obj.calc_margin(svc)
    obj.plot_points(size=svc_margin, show=False)
    obj.plot_svc_decision(svc, show=True)

    print 'Logit sum of margin:', logit_margin.sum()
    print 'SVC sum of margin:', svc_margin.sum()
