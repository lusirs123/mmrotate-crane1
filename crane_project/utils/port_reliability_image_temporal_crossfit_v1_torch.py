"""Train only small cached-feature quality heads; no detector imports/updates."""
import hashlib
import json
import math
import numpy as np
import torch
from crane_project.utils import port_reliability_image_temporal_crossfit_v1 as core
from crane_project.utils import port_reliability_feature_source_v1_torch as existing


def train_head(matrix,rows,indices,tag,device,stream):
    indices=np.asarray(indices,dtype=int)
    if indices.ndim!=1 or not len(indices) or len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=len(rows)):
        raise ValueError('Invalid fold fit indices')
    if any(rows[i]['reliability_role']!='train' or rows[i]['split'] not in ('train','train_sim') for i in indices):raise ValueError('TRAIN quality-head fitting only')
    part=np.array(matrix[indices],dtype=np.float64);normalizer=core.image.normalization(part)
    labels=np.array([int(core.temporal.base.level.metrics.bad(rows[i])) for i in indices],dtype=np.float32)
    if not labels.any() or labels.all():raise ValueError('Both genuine fold classes required')
    torch.manual_seed(core.SETTINGS['seed']);model=existing.ResidualRisk().to(device)
    count=sum(p.numel() for p in model.parameters())
    if count!=4290:raise ValueError('Quality capacity changed')
    initial=existing.exported(model);initial_sha=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
    x=torch.tensor(core.image.normalize(part,normalizer),dtype=torch.float32,device=device)
    z=torch.tensor(part[:,0],dtype=torch.float32,device=device)
    y=torch.tensor(labels,dtype=torch.float32,device=device)
    w=torch.tensor(core.image.class_weights(labels),dtype=torch.float32,device=device)
    with torch.no_grad():
        if not torch.equal(model(x,z),-z):raise ValueError('Quality zero initialization differs')
    opt=existing.optimizer(model);rng=np.random.RandomState(core.SETTINGS['seed']);updates=0;digest=hashlib.sha256()
    for epoch in range(1,core.SETTINGS['epochs']+1):
        order=rng.permutation(len(indices));digest.update(order.astype('<i8').tobytes())
        for slot,offset in enumerate(range(0,len(order),core.SETTINGS['batch_size'])):
            idx=torch.tensor(order[offset:offset+core.SETTINGS['batch_size']],dtype=torch.long,device=device)
            info=existing.update(model,opt,x[idx],z[idx],y[idx],w[idx]);updates+=1
            stream.write(json.dumps(dict(tag=tag,epoch=epoch,slot=slot,batch_count=len(idx),**info),allow_nan=False)+'\n')
        if epoch in (50,100):print('Quality head',tag,epoch,'/100',updates,flush=True)
    exported=existing.exported(model)
    with torch.no_grad():actual=torch.sigmoid(model(x,z)).cpu().numpy().astype(float)
    replay=core.temporal.base.simple.sigmoid(core.image.numpy_logits(exported,part,normalizer))
    error=float(abs(actual-replay).max())
    if error>1e-4:raise ValueError('Final head Torch/NumPy replay differs')
    return dict(model=exported,normalizer=normalizer,train_images=[rows[i]['image'] for i in indices],
                fit_indices=indices.tolist(),training_sequences=sorted(set(rows[i]['sequence'] for i in indices)),
                bad_rows=int(labels.sum()),good_rows=int(len(labels)-labels.sum()),
                final_epoch=100,updates=updates,batch_order_sha256=digest.hexdigest(),
                initial_sha256=initial_sha,parameter_count=count,risk_replay_maximum=error,
                dtype='float32',detector_updates=0)


def predict(head,matrix):
    return core.temporal.base.simple.sigmoid(core.image.numpy_logits(head['model'],matrix,head['normalizer']))
