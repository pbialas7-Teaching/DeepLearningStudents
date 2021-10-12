import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(ax, trajectory, n_batches=1, **kwargs):   
    epochs = np.linspace(0, len(trajectory)/n_batches,len(trajectory))
    ax.plot(epochs,trajectory[:,-1],'.', **kwargs)    

def plot_grad_and_trajectory(ax, trajectory, n_batches=1):   
    epochs = np.linspace(0, len(trajectory)/n_batches,len(trajectory))
    ax.plot(epochs,trajectory[:,-1],'.')
    tw = ax.twinx()
    tw.plot(epochs,np.linalg.norm(trajectory[:,:2], axis=1),'.r')

    l1,u1 = ax.get_ylim()
    l2,u2 = tw.get_ylim();
    nl2=(u2-l2)/(u1-l1)*l1
    dl2=nl2-l2
    ax.set_ylabel("MSE", color="blue")
    tw.set_ylim(l2+dl2,u2+dl2);
    tw.spines['right'].set_color('red')
    tw.tick_params(axis='y', colors='red')
    tw.set_ylabel("||grad||", color="red");
    ax.set_xlabel("epoch")