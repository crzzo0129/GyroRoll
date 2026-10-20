"""Initialize an ordinary trainable MLP from local LQR gains; no runtime controller."""
import json
from pathlib import Path
import torch


def affine_seed(dtype=torch.float32):
    data=json.loads(Path(__file__).with_name('linkage_seed_gains.json').read_text())
    gain=torch.tensor(data['K'],dtype=dtype)
    limit=torch.pi/6
    state=torch.zeros((17,20),dtype=dtype)
    state[1,7]=-1.;state[2,6]=1.  # Small-angle roll/pitch from projected gravity.
    state[4,10]=limit;state[5,13]=limit;state[6,12]=limit
    state[7,0]=1.;state[7,16]=-1.
    state[8,1]=1.;state[9,2]=1.
    state[10,3]=4.;state[11,4]=4.;state[12,5]=4.
    state[13,11]=10.;state[14,15]=10.;state[15,14]=10.
    state[16,9]=35.;state[16,16]=-1/.12
    scales=torch.tensor([35.,limit],dtype=dtype)
    weight=-(gain@state)/scales[:,None]
    weight[0,16]+=1/(.12*35.)
    bias=torch.tensor(data['equilibrium_ctrl'],dtype=dtype)/scales
    return weight,bias


def initialize_actor(actor, activation='elu'):
    """Positive ELU channels transmit raw observations; other channels remain learnable."""
    layers=[m for m in actor.modules() if isinstance(m,torch.nn.Linear)]
    assert layers[0].in_features==20 and layers[-1].out_features==2
    weight,bias=affine_seed()
    if activation=='relu':
        # Signed pairs represent Wx+b without a large cancelling output bias.
        with torch.no_grad():
            for i,layer in enumerate(layers[:-1]):
                layer.weight.zero_();layer.bias.zero_()
                if i==0:
                    eye=torch.eye(20,device=layer.weight.device)
                    layer.weight[:20,:20]=eye
                    layer.weight[20:40,:20]=-eye
                    torch.nn.init.normal_(layer.weight[40:],std=.2)
                else:
                    count=min(layer.in_features,layer.out_features)
                    layer.weight[:count,:count]=torch.eye(count,device=layer.weight.device)
            last=layers[-1]
            last.weight.zero_()
            last.weight[:,:20]=weight.to(last.weight.device)
            last.weight[:,20:40]=-weight.to(last.weight.device)
            last.bias.copy_(bias.to(last.bias.device))
        return
    if activation!='elu':
        raise ValueError(activation)
    with torch.no_grad():
        for i,layer in enumerate(layers[:-1]):
            layer.weight.zero_();layer.bias.zero_()
            layer.weight[:20,:20]=torch.eye(20,device=layer.weight.device)
            if i==0:
                layer.bias[:20]=5.
                torch.nn.init.normal_(layer.weight[20:],std=.2)
            else:
                remaining=min(layer.out_features,layer.in_features)-20
                layer.weight[20:20+remaining,20:20+remaining]=torch.eye(remaining,device=layer.weight.device)
        last=layers[-1]
        last.weight.zero_()
        last.weight[:,:20]=weight.to(last.weight.device)
        last.bias.copy_((bias-5*weight.sum(dim=1)).to(last.bias.device))


def grounded_height(roll,pitch,wheel_phase):
    """Only initial placement: tire's lowest capsule endpoint minus static compression."""
    angles=wheel_phase[:,None]+torch.arange(128,device=roll.device)[None,:]*(2*torch.pi/128)
    r22=roll.cos()*pitch.cos()
    ring_z=.108*(-pitch.sin()[:,None]*angles.sin()+r22[:,None]*angles.cos())
    return .3*r22-ring_z.min(dim=1).values+.012-.0005791213481721655
