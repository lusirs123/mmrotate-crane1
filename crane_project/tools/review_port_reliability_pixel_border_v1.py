#!/usr/bin/env python3
"""Independent offline label, identity, peak reduction and pair-count review."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parents[2]

def auc(bad,good):
    if not bad or not good:return None
    return sum((x>y)+.5*(x==y) for x in bad for y in good)/(len(bad)*len(good))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('result',type=Path);ap.add_argument('--name',default='server_review.json');a=ap.parse_args()
    report=json.loads((a.result/'report.json').read_text());rows=json.loads((a.result/'rows.json').read_text())
    protocol=json.loads((ROOT/'crane_project/tools/port_reliability_pixel_border_v1_protocol.json').read_text())
    for p,d in protocol['pins'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==d,p
    assert report['source_pins']==protocol['pins'] and report['settings']==protocol['settings']
    old={r['image']:r for r in map(json.loads,gzip.open(ROOT/protocol['rows'],'rt'))}
    assert len(rows)==2558 and len({r['image'] for r in rows})==2558 and set(old)=={r['image'] for r in rows}
    checks=0
    for r in rows:
        o=old[r['image']];assert r['pred']==o['pred'] and r['accepted']==o['fallback_decision']
        assert o['reliability_role']=='train' and o['split'] in ('train','train_sim')
        assert r['sequence']==o['sequence'] and r['domain']==o['domain']
        assert r['low_temporal']==(o['temporal']['available'] and o['temporal']['primary']<=.0785511443740936)
        if o['pred'] is not None:
            p=sorted(o['pred'][2:4],reverse=True);g=sorted(o['gt'][2:4],reverse=True)
            err=[abs(x-y)/y>.1 for x,y in zip(p,g)]
            assert r['wrong']==any(err)
            assert abs(r['scale']-math.sqrt(p[0]*p[1]))<1e-12
            assert r['error_type']==('both' if all(err) else 'L_only' if err[0] else 'S_only' if err[1] else 'correct')
            for axis in ('L','S'):
                e=r['evidence'][axis];supported=all(s['valid_profiles']>=5 for s in e['sides'])
                assert supported==e['supported']
                vals=[]
                for s in e['sides']:
                    assert len(s['peak_fractions'])==s['valid_profiles']
                    for p,z in zip(s['peak_fractions'],s['signed_log_offsets']):assert .25<=p<=.75 and abs(math.log(2*p)-z)<1e-12
                    vals+=s['signed_log_offsets']
                if supported:
                    assert abs(median([abs(x) for x in vals])-e['mismatch'])<1e-12
                    assert abs(median(vals)-e['signed_offset'])<1e-12
                    assert abs(median([abs(x-median(vals)) for x in vals])-e['dispersion'])<1e-12
                else:assert all(e[k] is None for k in ('mismatch','signed_offset','dispersion'))
        else:assert r['wrong'] is None and r['evidence'] is None and r['error_type']=='missing'
        checks+=1
    for name,g in report['result']['groups'].items():
        part=[r for r in rows if r['low_temporal'] and r['wrong'] is not None and (name=='all' or r[name.split(':')[0]]==name.split(':')[1])]
        assert g['n']==len(part) and g['wrong']==sum(r['wrong'] for r in part)
        assert g['error_types']=={k:sum(r['error_type']==k for r in part) for k in ('L_only','S_only','both','correct')}
        for axis in ('L','S'):
            v=[r for r in part if r['evidence'][axis]['supported']]
            b=[r['evidence'][axis]['mismatch'] for r in v if r['wrong']];good=[r['evidence'][axis]['mismatch'] for r in v if not r['wrong']]
            expect=auc(b,good);actual=g['axes'][axis]['pooled']['AUC_higher']
            assert expect==actual
            assert len(v)==g['axes'][axis]['supported'] and len(b)==g['axes'][axis]['pooled']['wrong'] and len(good)==g['axes'][axis]['pooled']['good']
            accepted=[r for r in v if r['accepted']]
            fa=[r['evidence'][axis]['mismatch'] for r in accepted if r['wrong']]
            cr=[r['evidence'][axis]['mismatch'] for r in accepted if not r['wrong']]
            ac=g['axes'][axis]['remaining_FA_vs_CR']
            assert ac['wrong']==len(fa) and ac['good']==len(cr) and ac['AUC_higher']==auc(fa,cr)
            wins=0.;n=0
            for x in v:
                for y in v:
                    if x['wrong'] and not y['wrong'] and x['sequence']==y['sequence'] and abs(math.log(y['scale']/x['scale']))<=math.log(1.1):
                        xx=x['evidence'][axis]['mismatch'];yy=y['evidence'][axis]['mismatch'];wins+=(xx>yy)+.5*(xx==yy);n+=1
            assert n==g['axes'][axis]['size_matched_pairs'] and (wins/n if n else None)==g['axes'][axis]['size_matched_auc']
            checks+=1
    qualified={axis:[name for name,g in report['result']['groups'].items() if name.startswith('sequence:real_') and g['axes'][axis]['pooled']['wrong']>=3 and g['axes'][axis]['pooled']['good']>=20 and g['axes'][axis]['size_matched_auc'] is not None and g['axes'][axis]['size_matched_auc']>=.55] for axis in ('L','S')}
    assert qualified==report['result']['qualifying_videos']
    assert report['result']['feasibility_passed']==any(report['result']['groups']['all']['axes'][axis]['pooled']['wrong']>=20 and len(qualified[axis])>=2 for axis in ('L','S'))
    assert not report['VAL_opened'] and not report['TEST_opened'] and report['fits']==0 and report['decisions_changed']==0 and report['detector_inferences']==0
    with (a.result/a.name).open('x') as f:json.dump(dict(passed=True,rows=len(rows),checks=checks,report_sha256=hashlib.sha256((a.result/'report.json').read_bytes()).hexdigest(),scope='independent labels identities peak reductions and AUC; image extractor covered by synthetic tests'),f,indent=2)
    print('REVIEW_PASS',len(rows),checks)
if __name__=='__main__':main()
