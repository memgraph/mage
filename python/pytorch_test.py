import mgp
import torch


@mgp.read_proc
def function(ctx:mgp.ProcCtx)->mgp.Record(output=mgp.List[int]):
    a = torch.zeros(1, 2)
    b = torch.ones(1, 2)
    print(b)
    print(a)
    c = torch.add(a,b)
    c = torch.flatten(c)
    print(c)
    d = c.numpy()
    print(d)
    return mgp.Record(output=[int(a) for a in d])
