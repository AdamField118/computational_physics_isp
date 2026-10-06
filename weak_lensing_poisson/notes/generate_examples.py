"""Reproduce the P1 mesh and assembly examples without a display server."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'figures'

def element_stiffness(vertices):
    affine=np.column_stack([np.ones(3),vertices])
    gradients=np.linalg.inv(affine)[1:,:].T
    return abs(np.linalg.det(affine))/2*gradients@gradients.T

def main():
    OUT.mkdir(exist_ok=True);n=5
    nodes=np.array([(j/n,i/n) for i in range(n+1) for j in range(n+1)])
    triangles=[]
    for i in range(n):
        for j in range(n):
            a=i*(n+1)+j;triangles.extend([[a,a+1,a+n+1],[a+1,a+n+2,a+n+1]])
    triangles=np.array(triangles);values=np.zeros(len(nodes));values[2*(n+1)+2]=1
    fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    axes[0].triplot(*nodes.T,triangles,color='0.4')
    im=axes[1].tripcolor(*nodes.T,triangles,values,shading='gouraud',cmap='viridis')
    for ax in axes:ax.set(aspect='equal',xlabel='x',ylabel='y')
    axes[0].set_title('36 nodes, 50 triangles');axes[1].set_title('Basis at (0.4, 0.4)')
    fig.colorbar(im,ax=axes[1],label='Basis value');fig.savefig(OUT/'mesh_basis.png',dpi=160);plt.close(fig)
    nodes=np.array([[.2,.2],[.8,.2],[.5,.8],[.2,.8],[.8,.8]])
    triangles=np.array([[0,1,2],[0,2,3],[1,4,2]])
    global_k=np.zeros((5,5));local=[];stages=[]
    for t in triangles:
        ke=element_stiffness(nodes[t]);local.append(ke.tolist());global_k[np.ix_(t,t)]+=ke;stages.append(global_k.copy())
    assert np.allclose(global_k,global_k.T) and np.allclose(global_k@np.ones(5),0)
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
    axes[0].triplot(*nodes.T,triangles,color='0.4')
    for i,v in enumerate(nodes):axes[0].text(*v,str(i),ha='center',va='bottom')
    axes[0].set(aspect='equal',title='Three-element mesh')
    for i,(ax,k) in enumerate(zip(axes[1:],stages),1):
        im=ax.imshow(k,vmin=-np.max(abs(global_k)),vmax=np.max(abs(global_k)),cmap='coolwarm')
        ax.set(title=f'After element {i}\n{np.count_nonzero(abs(k)>1e-10)} nonzeros',xticks=range(5),yticks=range(5))
    fig.colorbar(im,ax=list(axes[1:]),label='Stiffness entry');fig.savefig(OUT/'assembly.png',dpi=160);plt.close(fig)
    (OUT/'assembly_matrices.json').write_text(json.dumps({'nodes':nodes.tolist(),'triangles':triangles.tolist(),'element_matrices':local,'assembled_matrix':global_k.tolist()},indent=2)+'\n')

if __name__=='__main__':main()
