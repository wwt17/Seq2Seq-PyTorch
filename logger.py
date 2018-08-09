from matplotlib import pyplot as plt
plt.switch_backend('agg')

class LossLogger(object):
    def __init__(self, names, path):
        self.names = names
        if os.path.exists(path):
            with open(path, 'r') as f:
                names_ = tuple(f.readline().strip().split())
                assert self.names == names_, "given names: {} prev names: {}".format("\t".join(self.names), "\t".join(names_))
                self.a = [list(map(float, line.strip().split())) for line in f]
        else:
            with open(path, 'w') as f:
                print('\t'.join(names), file=f)
            self.a = []
        self.f = open(path, 'a', 1)
    def append(self, e):
        self.a.append(e)
        print('\t'.join(map(lambda x: "{:.6f}".format(x), e)), file=self.f)
    def recent(self, k):
        k = min(k, len(self.a))
        return list(map(np.mean, zip(*self.a[-k:])))
    def recent_repr(self, k):
        v = self.recent(k)
        return "\t".join("{}: {:.3f}".format(name, val) for name, val in zip(self.names, v))
    def plot(self, figure_name):
        plt.figure(figsize=(14, 10))
        shapes = ['x', '^', 'v', '--']
        sizes = [6, 2, 2, 2]
        for name, a, shape, size in zip(self.names, zip(*self.a), shapes, sizes):
            plt.plot(a, shape, linewidth=size, label=name)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.legend(loc=2., borderaxespad=0.)
        plt.savefig('{}.png'.format(figure_name))
        plt.close()

