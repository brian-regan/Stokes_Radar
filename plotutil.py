import os
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string
import seaborn as sns

FIGSIZE = (7, 3)
FONTSIZE = {'label': 7, 'tick': 6, 'annotation': 10, 'legend': 6}
TOP = 0.95
FIG_DIR = '../figs'

sns.set_style("ticks")


class Figure:
    def __init__(self, h, w=None, figsize=FIGSIZE, fontsize=FONTSIZE,
                 annotation=False):
        if w is None:
            length = h
            h = math.floor(math.sqrt(length))
            w = math.ceil(length/h)
        self.fig = plt.figure(figsize=figsize)
        self.fontsize = fontsize
        self.annotation = annotation
        self.gs = gridspec.GridSpec(h, w)
        self.axes = [self.fig.add_subplot(grid) for grid in self.gs]
        for i, axis in enumerate(self.axes):
            axis.tick_params(labelsize=fontsize['tick'])
            if annotation:
                axis.text(-0.2, 1.1, '({})'.format(string.ascii_lowercase[i]),
                          transform=axis.transAxes,
                          size=fontsize['annotation'])

    def __getitem__(self, n):
        return self.axes[n]

    def add_label(self, n, xlabel=None, ylabel=None, title=None):
        axis = self[n]
        if xlabel:
            axis.set_xlabel(xlabel, fontsize=self.fontsize['label'])
        if ylabel:
            axis.set_ylabel(ylabel, fontsize=self.fontsize['label'])
        if title:
            axis.set_title(title, fontsize=self.fontsize['label'])

    def save(self, dirname, filename=None):
        if filename is None:
            filename = dirname
            dirname = FIG_DIR
        if self.annotation:
            self.gs.tight_layout(self.fig, rect=(0, 0, 1, TOP))
        else:
            self.fig.tight_layout()
        sns.despine()
        plt.savefig(os.path.join(dirname, filename + '.eps'))

    def close(self, dirname, filename=None):
        self.save(dirname, filename)
        plt.close()
