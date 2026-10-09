"""GT-free native B feature observer on the original frozen TEST pipeline."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from crane_project.tools import run_port_reliability_feature_source_v1 as parent
ROOT=parent.ROOT
prior=parent.prior

def collect(rows, gpu, out):
    role = "TEST"
    import torch
    from mmcv.parallel import collate,scatter
    from crane_project.tools import train_port_geometry_midpoint_formal_v1 as frozen
    from crane_project.utils import port_reliability_feature_source_v1_torch as native
    from crane_project.tools import eval_port_geometry_midpoint_v1_test as test
    cfg = frozen.g.check_cfg(); sources,data_identity = test.fixed_test_inputs()
    source = {r['image']:r for r in sources}; lookup = {r['image']:r for r in rows}
    detector = frozen.build_detector(cfg,gpu); before = frozen.g.state_digest(detector)
    cache = json.loads((ROOT/prior.CACHE/'cache_manifest.json').read_text())
    if before != cache['detector_state']: raise ValueError('Frozen B state differs from source cache')
    dataset = test.build_test_dataset(cfg,sources)
    names = [Path(info['filename']).stem for info in frozen.infos(dataset)]
    if len(names) != len(lookup) or set(names) != set(lookup): raise ValueError('Full dataset membership differs')
    captured = {}; maximum = np.zeros(6); traces = []
    for index,name in enumerate(names):
        row = lookup[name]
        if source[name]['image_sha256'] != row['image_sha256'] or source[name]['annotation_sha256'] != row['annotation_sha256']:
            raise ValueError('Frozen TEST bytes differ: '+name)
        value = scatter(collate([dataset[index]],samples_per_gpu=1),[gpu])[0]
        image,metas = value['img'][0],value['img_metas'][0]
        if (len(value['img']) != 1 or image.shape != (1,3,1024,1024) or len(metas) != 1
                or Path(metas[0]['filename']).stem != name or metas[0].get('flip',False)
                or list(metas[0]['ori_shape'][:2][::-1]) != row['image_size']):
            raise ValueError('Deterministic native view differs: '+name)
        frozen.g.checked_meta(metas[0]); frozen.g.assert_detector_frozen(detector)
        prediction,feature,trace = native.trace_native(detector,image,metas)
        b = frozen.g.flatten_prediction(prediction)
        if len(b) != int(row['pred'] is not None) or (feature is None) != (not len(b)):
            raise ValueError('Native B/M output/missing identity differs: '+name)
        if len(b):
            stored = np.asarray(row['b_original']); delta = np.abs(b[0]-stored)
            delta[4] = abs((b[0,4]-stored[4]+np.pi/2)%np.pi-np.pi/2)
            maximum = np.maximum(maximum,delta)
            if (not np.allclose(b[0,:4],stored[:4],atol=5e-4,rtol=2e-5)
                    or delta[4]>2e-6 or b[0,5] != stored[5]):
                raise ValueError('Native original B replay differs: '+name)
            captured[name] = feature
            trace['patch_sha256'] = hashlib.sha256(feature.astype('<f4').tobytes()).hexdigest()
        traces.append(dict(image=name,role=role,image_sha256=source[name]['image_sha256'],**trace))
        if index%50 == 0 or index==len(names)-1: print('Native source',role,index+1,'/',len(names),flush=True)
        del value,image,metas,prediction,b
    if frozen.g.state_digest(detector) != before: raise ValueError('Detector state changed')
    ids = sorted(captured); raw = np.stack([captured[i] for i in ids])
    np.savez_compressed(out/(role.lower()+'_native_raw.npz'),features=raw,images=np.asarray(ids))
    with gzip.open(out/(role.lower()+'_native_trace.jsonl.gz'),'xt') as stream:
        for r in sorted(traces,key=lambda r:r['image']): stream.write(json.dumps(r,allow_nan=False)+'\n')
    proof = dict(frames=len(rows),outputs=len(ids),data_identity=data_identity,detector_state=before,
        maximum_original_B_replay_difference=maximum.tolist(),actual_native_topk=True,
        new_detector_inferences=len(rows),detector_updates=0,GT_online=False)
    del detector,dataset; torch.cuda.empty_cache()
    return raw,ids,proof

