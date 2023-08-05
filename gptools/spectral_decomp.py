from matplotlib import pyplot as plt
import numpy as np

f = 1
l = np.pi/4
N = 5

get_mode = lambda n: (lambda x: np.exp(1j*n*x))

# some kernel
k_dx = lambda dx: f * np.exp(-dx ** 2 / (2 * l ** 2))

# attenuating distribution for proper kernel action
p = lambda x: 1/(2*np.pi) # measure of circle, i.e. circumference

def k(x1, x2):
    x1 = np.atleast_2d(x1).T
    x2 = np.atleast_2d(x2).T
    return k_dx(np.linalg.norm(x1 - x2, axis=-1))

dths = np.linspace(-np.pi,np.pi,5001)


mode = lambda k: (lambda x: (2*np.pi)**-0.5*np.cos(k/2*x))

# get F coefficients
def get_aks(k_dx, N):
    ns = np.arange(N)
    ks = np.concatenate([-ns, ns[1:]])
    circ = np.linspace(-np.pi, np.pi, 5001)
    dth = np.diff(circ)[0]
    a_ks = {k: (2*np.pi)**-0.5*np.sum(k_dx(circ) * np.exp(-1j*k*circ))*dth for k in ks}
    for k in ks:
        assert np.allclose(a_ks[k], a_ks[-k], atol=1e-3)
    return a_ks


def get_k_trunc(a_ks, omit_neg=False):
    # if omit_neg:
    #     a_ks = {k:v*(0 if omit_neg and v < 0 else 1) for k, v in a_ks.items()}
    # def trunc_kernel(dth):
    #     v = np.sum([((a_ks[k]*(mode(k)(dth)) + a_ks[-k]*(mode(-k)(dth))) if k != 0 else a_ks[0]*(mode(0)(dth))) for k in range(n+1)], axis=0)
    #     return v

    def trunc_kernel(dth):
        v = np.sum([a_ks[k]*(mode(k)(dth)) for k in range(len(a_ks))], axis=0)
        return v

    return trunc_kernel
    
if __name__ == "__main__":

    # plot the spectrum
    a_ks = get_aks(k_dx, N)
    axx = plt.gca()
    axx.plot(dths, k_dx(dths), c="C0", ls="--", zorder=100, lw=3)
    axf = axx.twiny()
    ns = np.arange(N)
    ks = np.concatenate([-ns, ns[1:]])
    axf.plot(ks,[ a_ks[k] for k in ks], c="C1", ls="none", marker="o")

    for ik, k in enumerate(range(N)):
        k_trunc = get_k_trunc(a_ks, ik)
        axx.plot(dths, k_trunc(dths), label=f"reconstruction {ik}", c=plt.get_cmap("Greys")(ik/len(ks[ks>0])))

    # axx.legend()
    plt.savefig("test.png")