import matplotlib.pyplot as plt


class Plot(object):
    def __init__(self, refresh_time=0.01):
        _, self.fig = plt.subplots()
        self.refresh_time = refresh_time
        plt.ion()
        plt.show()

    def set_label_and_title(self, x_label=None, y_label=None, title=None):
        self.fig.set_xlabel(x_label)
        self.fig.set_ylabel(y_label)
        self.fig.set_title(title)

    def add_plot(self, x_data, y_data, color, label=None, **kw_args):
        self.fig.cla()
        self.fig.plot(x_data, y_data, label=label, c=color)
        self.set_label_and_title(**kw_args)
        plt.pause(self.refresh_time)
        plt.legend()
