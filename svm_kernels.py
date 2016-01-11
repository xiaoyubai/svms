import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

class SVM_kernels(object):
    def __init__(self, df, xnames, yname):
        self.df = df
        self.xnames = xnames
        self.yname = yname
        self.X_scaled = None
    def fit(self, X, y, kernel, C=None):
        if C:
            pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=C, kernel=kernel))])
        else:
            pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel=kernel))])
        self.X = X
        self.y = y
        self.X_scaled = pipeline.named_steps['scaler'].fit_transform(self.X)
        pipeline.fit(self.X_scaled, self.y)
        return pipeline

    def plot_points(self, size=None, show=False):

        if size == None:
            plt.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], c=self.y)
        else:
            plt.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], c=self.y, s=size)
        plt.xlabel(self.xnames[0])
        plt.ylabel(self.xnames[1])
        if show:
            plt.show()

    def find_C(self, kernel, cv=5, show=True):
        scores = []
        C_range = np.linspace(0.01, 1, 200)
        for C in C_range:
            pipeline = self.fit(kernel=kernel, C=C)
            pipeline.fit(self.X_scaled, self.y)
            score = np.mean(cross_val_score(pipeline, self.X_scaled, self.y, cv=cv))
            scores.append(score)
        plt.plot(C_range, scores)
        score_min = min(scores) - np.std(scores)
        score_max = max(scores) + np.std(scores)
        plt.ylim(score_min, score_max)
        if show:
            plt.show()
        return C_range[np.argmax(scores)], max(scores)

    def plot_svc_decision(self, svc, show=True):
        # get the separating hyperplane
        w = svc.coef_[0]
        a = -w[0] / w[1]
        x = self.X_scaled[:, 0]
        y = self.X_scaled[:, 1]
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
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        if show:
            plt.show()

    def decision_boundary(self, clf, X, Y, h=.02):
        """Inputs:
            clf - a trained classifier, with a predict method
        """
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = self.X_scaled[:, 0].min() - .5, self.X_scaled[:, 0].max() + .5
        y_min, y_max = self.X_scaled[:, 1].min() - .5, self.X_scaled[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

        # Plot also the training points
        plt.scatter(self.X_scaled[:, 0], self.X_scaled[:, 1], c=self.Y, edgecolors='k', cmap=plt.cm.Paired)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def get_accuracy(self, X, y, kernel, cv=5, plot=True):
        scores = []
        pipeline = Pipeline([('scaler', StandardScaler()),
                        ('model', SVC(kernel=kernel))])
        pipeline.fit(X, y)
        if plot:
            self.decision_boundary(pipeline.named_steps['model'], X, y, h=.02)
        score = np.mean(cross_val_score(pipeline, X, y, cv=5))
        scores.append(score)
        return np.mean(scores)
