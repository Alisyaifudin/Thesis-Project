import numpy as np
import vaex
from os.path import join, abspath, dirname
import pathlib
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

current = pathlib.Path(__file__).parent.resolve()
root_dir = join(current, "..",)

root_data_dir = join(root_dir, "Data")
data_dir = join(root_data_dir, "Completeness")

class compjk:
    def __init__(self, comp_path: str):
        df_args = vaex.open(comp_path)
        df_args = df_args.to_pandas_df()
        # jk_lows = df_args.jk_low.values
        self.jk_highs = df_args.jk_high.values
        x0s = df_args.x0.values
        self.x0s = np.append(x0s, 0)
        ws = df_args.w.values
        self.ws = np.append(ws, 1)
        ps = df_args[["p1", "p2", "p3", "p4", "p5"]].values
        self.ps = np.append(ps, np.zeros((len(ps), 5)), axis=0)
    def window(self, x, x0, w, n):
        return 1/(1+np.exp(((x-x0)/w)**(2*n)))*2
    def polyv(self, p, j):
        print(p.shape)
        print(j.shape)
        return np.polyval(p, j)
    def interp(self, j, jk):
        index = np.searchsorted(self.jk_highs, jk)
        x0 = self.x0s[index]
        w = self.ws[index]
        p = self.ps[index]
        polifit_v = np.array([np.polyval(pi, ji) for pi, ji in zip(p, j)])
        return np.select([jk > 0.4, jk < -0.1], [0, 0], default=self.window(j, x0, w, 18)*polifit_v)

