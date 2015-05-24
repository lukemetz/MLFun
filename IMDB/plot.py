import os
import matplotlib.pylab as plt
from blocks.extensions import SimpleExtension

class PlotHistogram(SimpleExtension):
    def __init__(self, channels, bins=20, path='.', prefix=None, **kwargs):
        super(PlotHistogram, self).__init__(**kwargs)
        self.channels = channels
        self.prefix = prefix
        self.path = path
        self.bins = bins

    def do(self, which_callback, *args):
        log = self.main_loop.log
        for c in self.channels:
            if c in log.current_row:
                data = log.current_row[c]
                plt.figure()
                plt.hist(data.ravel(), bins=self.bins)
                plt.title(c)

                if self.prefix:
                    file_name = "%s_%s.png"%(self.prefix, c)
                else:
                    file_name = "%s.png"%c
                plt.savefig(os.path.join(self.path, file_name))

if __name__ == "__main__":
    from blocks.main_loop import MainLoop
    import numpy as np

    class Mock(object):
        pass

    p = PlotHistogram(channels=['test'])
    p.main_loop = Mock()
    p.main_loop.log = Mock()
    p.main_loop.log.current_row = dict(test=np.random.normal(size=(10, 10)))
    p.do("unused")
