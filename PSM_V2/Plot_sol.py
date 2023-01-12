import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcdefaults()
matplotlib.rcParams.update({'font.size': 15})
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import numpy as np
class Plot:
    def __init__(self, test_xs, gt, model, data):
        self.model = model
        print(self.model)
        self.gt= gt
        self.test_xs = test_xs
        self.data = data
        x = self.test_xs[0]
        y = self.test_xs[1]
        Xt, Yt = np.meshgrid(self.test_xs[0],self.test_xs[1])
        out = self.model(data).T[0].reshape(249,249).detach().numpy()
        u_sol = self.gt(Xt,Yt)
        #out = N_p()._eval(X_r).reshape(100,100)
        L0_inf = np.max(abs(out-u_sol))
        #Lp_inf = torch.max(abs(poisson_residual(net_s(inp_r),inp_r,omega).reshape(-1)))
        L0_mean =np.mean(abs(out-u_sol))
        print("pred rel. linf-error = {:e}".format(L0_inf))
        print("pred rel. l2-error = {:e}".format(L0_mean))
        #print("pde res. linf-error = {:e}".format(Lp_inf))
        xl = np.linspace(-1,1, 5)
        x_ticks = np.around(xl, 1)
        plt.subplot(1,3,1)
        plt.imshow(u_sol, cmap="Spectral", origin="lower")
        #plt.clim(-0.3,0.4)
        plt.xticks(np.linspace(0,len(x), 5, dtype = int),x_ticks)
        plt.yticks(np.linspace(0,len(x), 5, dtype = int),x_ticks)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title("Ground Truth")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1,3,2)
        plt.imshow(out, cmap="Spectral", origin="lower")
        plt.xticks(np.linspace(0,len(x), 5, dtype = int),x_ticks)
        plt.yticks(np.linspace(0,len(x), 5, dtype = int),x_ticks)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title("Prediction")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1,3,3)
        plt.imshow(np.abs(out-u_sol), cmap="Spectral", origin="lower")
        #plt.clim(np.min(np.abs(out-u_sol)/np.max(np.abs(u_sol))),np.max(np.abs(out-u_sol)/np.max(np.abs(u_sol))))
        plt.xticks(np.linspace(0,len(x), 5, dtype = int),x_ticks)
        plt.yticks(np.linspace(0,len(x), 5, dtype = int),x_ticks)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title("Point-wise Error")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.gcf().set_size_inches(14,4)
        plt.tight_layout()
        #äplt.savefig(folder + 'L_inf_error.png')
        #plt.savefig(folder + 'pred_error_MSE.png',bbox_inches='tight')
        