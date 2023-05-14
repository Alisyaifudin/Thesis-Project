from matplotlib import pyplot as plt

def style(name="seaborn-v0_8-deep", tex=True):
    plt.style.use(name)
    params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : tex,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern"]}
    plt.rcParams.update(params)