"""Reproduce the static Chapter 3 and 4 examples with NumPy and Matplotlib."""
from pathlib import Path
from math import comb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT=Path(__file__).resolve().parent

def save(fig, chapter, name):
    out=ROOT/f'chapter_{chapter}'/'figures'/f'{name}.png'
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160,bbox_inches='tight');plt.close(fig)

def main():
    x,y=np.meshgrid(np.linspace(-1,1,80),np.linspace(0,1,60))
    vals=[(1-x)*(1-y)/2,(1+x)*(1-y)/2,(1+x)*y/2,(1-x)*y/2]
    assert np.allclose(sum(vals),1)
    fig,axes=plt.subplots(2,2,figsize=(8,6),constrained_layout=True)
    for i,(ax,z) in enumerate(zip(axes.flat,vals),1):
        im=ax.pcolormesh(x,y,z,vmin=0,vmax=1,cmap='viridis',shading='auto')
        ax.set(title=f'Basis {i}',xlabel='x',ylabel='y',aspect='equal')
    fig.colorbar(im,ax=axes,label='Basis value');save(fig,3,'rectangle')
    pts=np.array([(i/40,j/40) for i in range(41) for j in range(41-i)])
    x,y=pts.T;tri=mtri.Triangulation(x,y);l=[1-x-y,x,y]
    vals=[v*(2*v-1) for v in l]+[4*l[0]*l[1],4*l[1]*l[2],4*l[0]*l[2]]
    assert np.allclose(sum(vals),1)
    fig,axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
    for i,(ax,z) in enumerate(zip(axes.flat,vals),1):
        im=ax.tripcolor(tri,z,vmin=-.125,vmax=1,cmap='viridis',shading='gouraud')
        ax.set(title=f'Basis {i}',xlabel='x',ylabel='y',aspect='equal')
    fig.colorbar(im,ax=axes,label='Basis value');save(fig,3,'quadratic_triangle')
    v=np.array([(i/2,j/2) for j in range(3) for i in range(3)])
    t=np.array([[0,1,3],[1,4,3],[1,2,4],[2,5,4],[3,4,6],[4,7,6],[4,5,7],[5,8,7]])
    edges=sorted({tuple(sorted((a,b))) for row in t for a,b in zip(row,np.roll(row,-1))})
    assert len(edges)==16
    mids=np.array([(v[a]+v[b])/2 for a,b in edges])
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,p,title in zip(axes,[v,mids],['Conforming: 9 vertex DOFs','Crouzeix–Raviart: 16 edge DOFs']):
        ax.triplot(*v.T,t,color='0.65');ax.scatter(*p.T,color='#0072B2',zorder=3)
        ax.set(title=title,aspect='equal',xlabel='x',ylabel='y')
    save(fig,3,'nonconforming')
    fig=plt.figure(figsize=(15,6),layout='constrained')
    tetra=np.array([[0,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],[.5,np.sqrt(3)/6,np.sqrt(2/3)]])
    for r in range(1,6):
        b=np.array([(i,j,r-i-j) for i in range(r+1) for j in range(r+1-i)])/r
        assert len(b)==comb(r+2,2)
        ax=fig.add_subplot(2,5,r);ax.plot([0,1,0,0],[0,0,1,0],color='0.7');ax.scatter(b[:,1],b[:,2],s=16)
        ax.set(title=f'r={r}: {len(b)} nodes',aspect='equal');ax.set_axis_off()
        b=np.array([(i,j,k,r-i-j-k) for i in range(r+1) for j in range(r+1-i) for k in range(r+1-i-j)])/r
        assert len(b)==comb(r+3,3)
        ax=fig.add_subplot(2,5,r+5,projection='3d')
        for i in range(4):
            for j in range(i):ax.plot(*tetra[[i,j]].T,color='0.7',lw=.8)
        ax.scatter(*(b@tetra).T,s=12,color='#D55E00');ax.set_title(f'{len(b)} tetrahedron nodes');ax.set_axis_off()
    save(fig,3,'lagrange_nodes')
    g=np.array([[-1,-1],[1,0],[0,1]]);k=.5*g@g.T
    assert np.allclose(k@np.ones(3),0)
    fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    axes[0].plot([0,1,0,0],[0,0,1,0],color='0.3')
    for i,grad in enumerate(g):
        axes[0].arrow(1/3,1/3,grad[0]*.3,grad[1]*.3,head_width=.035,length_includes_head=True)
        axes[0].text(1/3+grad[0]*.36,1/3+grad[1]*.36,f'grad {i+1}',ha='center')
    axes[0].set(aspect='equal',title='Reference gradients',xlim=(-.15,1.1),ylim=(-.15,1.1));axes[1].imshow(k,cmap='coolwarm',vmin=-1,vmax=1)
    for i in range(3):
        for j in range(3):axes[1].text(j,i,f'{k[i,j]:g}',ha='center',va='center')
    axes[1].set(title='Element stiffness',xticks=range(3),yticks=range(3));save(fig,4,'reference_stiffness')
    shapes=[[[0,0],[1,0],[.5,np.sqrt(3)/2]],[[0,0],[1,0],[0,1]],[[0,0],[1,0],[.2,.1]],[[0,0],[1,0],[.4,.015]]]
    fig,axes=plt.subplots(1,4,figsize=(12,3),constrained_layout=True)
    for ax,vs,label in zip(axes,shapes,['Equilateral','Right','Thin','Nearly collinear']):
        v=np.array(vs);sides=np.linalg.norm(v-np.roll(v,1,axis=0),axis=1);s=sides.sum()/2;area=np.sqrt(s*np.prod(s-sides));rho=area/s
        angles=[]
        for i in range(3):
            a=v[(i+1)%3]-v[i];b=v[(i+2)%3]-v[i]
            angles.append(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
        ax.fill(*np.vstack([v,v[0]]).T,facecolor='#56B4E9',edgecolor='0.2')
        ax.set(aspect='equal',xlim=(-.1,1.1),ylim=(-.1,1.1),title=f'{label}\nmin angle={min(angles):.2f}°\nh/rho={max(sides)/rho:.2f}')
    save(fig,4,'mesh_quality')
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for s in [.5,1,2]:axes[0].plot(np.array([0,1,0,0])*s,np.array([0,0,1,0])*s,label=f's={s}')
    axes[0].set(aspect='equal',title='Isotropic dilation');axes[0].legend();ss=np.geomspace(.1,3,80)
    for values,label in [(ss,'L2'),(np.ones_like(ss),'H1 seminorm'),(1/ss,'H2 seminorm')]:axes[1].loglog(ss,values,label=label)
    axes[1].set(xlabel='Scale s',ylabel='Norm ratio',title='Two-dimensional scaling');axes[1].legend();save(fig,4,'homogeneity')
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    n=16;h=1/(n+1);eig=4/h*np.sin(np.arange(1,n+1)*np.pi/(2*(n+1)))**2
    axes[0].bar(range(1,n+1),eig);axes[0].set(xlabel='k',ylabel='Eigenvalue',title='n=16 interior nodes')
    ns=np.array([4,8,16,32,64,128]);hs=1/(ns+1);cond=1/np.tan(np.pi/(2*(ns+1)))**2
    axes[1].loglog(hs,cond,'o-',label='Exact spectrum');axes[1].loglog(hs,4/(np.pi**2*hs**2),'--',label='4 / (pi² h²)')
    axes[1].set(xlabel='h',ylabel='Condition number');axes[1].legend();save(fig,4,'conditioning')
    quads=[[[0,0],[1,0],[1,1],[0,1]],[[0,0],[1,0],[1.4,1],[.4,1]],[[0,0],[1,0],[.8,1],[.2,1]],[[0,0],[1.3,.1],[1,1.1],[-.2,.8]],[[0,0],[1,0],[.3,.3],[0,1]]]
    def mapped(x,y,v):return np.stack([(1-x)*(1-y),(1+x)*(1-y),(1+x)*(1+y),(1-x)*(1+y)],axis=-1)@v/4
    fig,axes=plt.subplots(1,5,figsize=(15,3.5),constrained_layout=True)
    for ax,vs,label in zip(axes,quads,['Square','Parallelogram','Trapezoid','General convex','Concave']):
        v=np.array(vs);grid=np.linspace(-1,1,6)
        for t in grid:
            ax.plot(*mapped(np.full(30,t),np.linspace(-1,1,30),v).T,color='#0072B2',lw=.8)
            ax.plot(*mapped(np.linspace(-1,1,30),np.full(30,t),v).T,color='#0072B2',lw=.8)
        dets=[]
        for x in grid:
            for y in grid:
                dx=np.array([-(1-y),1-y,1+y,-(1+y)])@v/4;dy=np.array([-(1-x),-(1+x),1+x,1-x])@v/4
                dets.append(np.linalg.det(np.column_stack([dx,dy])))
        ax.set(aspect='equal',title=f'{label}\ndet J: {min(dets):.3f} to {max(dets):.3f}')
    save(fig,4,'quadrilateral_mapping')

if __name__=='__main__':main()
