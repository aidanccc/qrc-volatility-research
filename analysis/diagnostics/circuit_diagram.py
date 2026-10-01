"""Compact circuit drawing, with gate counts extracted from the actual builder."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from quantum_reservoir_trotter import build_trotter_step


def draw_gate_sequence(couplings):
    step=build_trotter_step(10,couplings,1.)
    gates=[]
    for i,instruction in enumerate(step.data):
        gates.append({'order':i+1,'gate':instruction.operation.name,
                      'qubits':','.join(str(step.find_bit(q).index) for q in instruction.qubits),
                      'angle_radians':float(instruction.operation.params[0])})
    assert step.count_ops()=={'rxx':45,'rz':10}
    fig,ax=plt.subplots(figsize=(12,6.5));ax.axis('off');ax.set_xlim(-2,12);ax.set_ylim(-1.2,11.5)
    def box(x,y,w,h,label,color='#e6edf2'):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#40505c',lw=1.1,zorder=3))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,zorder=4)
    for q in range(10):
        y=9-q
        ax.plot([0,10.8],[y,y],color='#929da5',lw=1,zorder=1)
        ax.text(-.15,y,f'q{q}',ha='right',va='center')
        if q<7:box(.4,y-.3,1.35,.6,'RY(πx)', '#e0eef1')
        box(6.1,y-.3,1.4,.6,'RZ(2Δt)')
        box(9.3,y-.3,.65,.6,'M','#e0eef1')
    box(3.2,-.35,2.0,9.7,'45 × RXX\none per pair')
    ax.text(4.2,10,'RXX angle = 2JᵢⱼΔt',ha='center',fontsize=10)
    ax.plot([2.9,2.9,7.8,7.8],[10.35,10.65,10.65,10.35],c='#40505c',lw=1)
    ax.text(5.35,11,'Repeat n times',ha='center',fontsize=11)
    ax.text(1.08,-.85,'Encode input',ha='center',fontsize=10)
    ax.text(9.65,-.85,'Final readout',ha='center',fontsize=10)
    ax.text(-1.3,6,'7 inputs',rotation=90,ha='center',va='center',fontsize=10)
    ax.text(-1.3,1,'3 memory',rotation=90,ha='center',va='center',fontsize=10)
    ax.plot([-1,-1],[2.5,9.35],color='#929da5');ax.plot([-1,-1],[-.35,2.35],color='#929da5')
    return fig,pd.DataFrame(gates)
