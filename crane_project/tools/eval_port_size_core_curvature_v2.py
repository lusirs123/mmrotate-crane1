"""Internal paired epoch04 evaluator, called only by new runner assess mode."""
import json
import time
from crane_project.utils import port_size_core_curvature_v2 as core
from crane_project.utils import port_size_core_comparison_v2 as comparison


def run(args,prepared):
    from crane_project.tools import run_port_size_core_curvature_v2 as entry
    old,contract=prepared
    trained=entry.stage(args.train_report,entry.STATUSES['train'],contract)
    formal,torch,detector,head,pipeline,arms,initial=entry.models(args,old)
    if initial != trained['initial_sha256']: raise ValueError('Initial identity differs')
    before=dict(b=entry.base.state_digest(detector),midpoint=entry.base.state_digest(head))
    if before!=trained['frozen']: raise ValueError('Candidate and assessment front end differ')
    for arm,model in arms.items():
        item=trained['checkpoints'][arm]
        path=args.train_report.parent/item['path']
        if path.parent != args.train_report.parent/arm or path.name!='epoch_04.pth':
            raise ValueError('Only fixed per-arm epoch04 allowed')
        if entry.base.sha(path)!=item['sha256'] or entry.base.sha(str(path)+'.sha.json')!=item['marker_sha256']:
            raise ValueError('Paired checkpoint index differs')
        payload=entry.load(path,contract,'paired_reference_train',arm,torch)
        if (payload['epoch']!=4 or payload['steps']!=1536 or payload['frozen']!=before
                or payload['initial_sha256']!=initial): raise ValueError('Budget/source identity differs')
        model.load_state_dict(payload['state'],strict=True);model.eval().requires_grad_(False)
    before.update({a:entry.base.state_digest(m) for a,m in arms.items()})
    runtime=entry.migration.new.MidpointReliability(old[2],old[1][5]['front_end'])
    raw=old[1][1][7];records=[];delta={'b':0.,'midpoint':0.};start=time.monotonic()
    with (args.out_dir/'paired_assessment_rows.jsonl').open('x') as stream:
        for index,(role,row) in enumerate(old[3],1):
            features,meta,_=entry.reference.view(row,raw,detector,pipeline,args.gpu)
            first=entry.migration.midpoint_from_features(formal,torch,detector,head,features,meta)
            for key,field in (('b','b_original'),('midpoint','pred')):
                difference=entry.parent.assessment.paired_prediction(row[field],first[key],row['image'])
                delta[key]=max(delta[key],difference)
            evidence={}
            for arm,model in arms.items():
                with torch.no_grad():probability=model(features[0]).sigmoid()[0,0].cpu().numpy()
                evidence[arm]=core.online_reading(probability,row['pred'],row['image_size'],meta,runtime,old[1][0])
            after=entry.migration.midpoint_from_features(formal,torch,detector,head,features,meta)
            if first!=after: raise ValueError('Paired evidence changed frozen final output')
            item=comparison.annotate(row,evidence,role)
            item['final_output_before_after_exact']=True
            item['transform_meta']={k:list(meta[k]) for k in ('img_shape','ori_shape','pad_shape','scale_factor')}
            item=json.loads(json.dumps(item,default=lambda v:v.item(),allow_nan=False))
            stream.write(json.dumps(item,allow_nan=False)+'\n');records.append(item)
            del features,probability
            if index%100==0:
                stream.flush();print('Paired fixed assessment',role,index,'/',len(old[3]),flush=True)
    after=dict(b=entry.base.state_digest(detector),midpoint=entry.base.state_digest(head),
               **{a:entry.base.state_digest(m) for a,m in arms.items()})
    if before!=after: raise ValueError('Assessment modified model states')
    restored=[json.loads(line) for line in (args.out_dir/'paired_assessment_rows.jsonl').read_text().splitlines()]
    stats=comparison.summarize(records)
    if restored!=records or stats!=comparison.summarize(restored): raise ValueError('Saved metrics do not replay exactly')
    report=dict(status=entry.STATUSES['assess'],contract=contract,train_report_sha256=entry.base.sha(args.train_report),
        **stats,state_before=before,state_after=after,max_absolute_formal_box_difference=delta,
        frozen_boxes_scores_output_count_center_and_angle_flags_preserved=True,GT_online=False,
        elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(args.gpu)/2**20,test_read=False,
        evidence_role='bounded_reference_holdout_and_exposed_source_VAL_not_independent_confirmation',
        reference_accuracy_ranking_and_workpoint_must_be_judged_separately=True,
        continuation_criteria='handoff_21.6_R_Q1_Q2_comparison_not_final_deployment_acceptance',
        automatic_performance_PASS=False)
    entry.base.write_new(args.out_dir/'assessment.json',report)
    for domain in ('real','sim'):
        group=stats['summary']['val']['domain:'+domain]
        print(domain,'95% comparison bound',group['comparison_95pct_acceptance_bound'],flush=True)
        print(domain,'same-count FA/FR',{m:(v['FA'],v['FR']) for m,v in group['primary_same_count'].items()},flush=True)
    return report['status']
