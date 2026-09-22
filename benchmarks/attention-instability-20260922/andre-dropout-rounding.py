import torch,json
from torch.nn.attention import sdpa_kernel,SDPBackend
out=[]
for p in [0.,.1,.125,.25,.5]:
 torch.manual_seed(42)
 q=torch.zeros(2,8,512,96,device='cuda',dtype=torch.bfloat16);q[...,0]=100;q.requires_grad_()
 k=torch.zeros_like(q);k[:,:,0,0]=100;k.requires_grad_();v=torch.randn_like(q,requires_grad=True);dy=torch.randn_like(q)
 with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):y=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=p)
 (y*dy).sum().backward();r={'dropout':p,'q_grad_norm':q.grad.float().norm().item(),'k_grad_norm':k.grad.float().norm().item()};out.append(r);print(json.dumps(r))
open('/opt/dlami/nvme/andre/dropout-rounding.json','w').write(json.dumps(out,indent=2))
