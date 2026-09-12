#!/usr/bin/env python3
"""Finalize true-online focused-paper output with post-inference GT metrics."""

import argparse
import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from crane_project.tools import base_v3_obb_hybrid_fixed_test_diagnostic_v51 as diagnostic
from crane_project.tools.base_v3_obb_component_reliability_continuous import (
    COMPONENTS, _error_from_value, _periodic_error_deg)
from crane_project.tools.base_v3_obb_external_sequence_eval import state_report
from crane_project.tools.base_v3_obb_fixed_test_eval import (
    ATTRIBUTION_PROTOCOL, build_records)
from crane_project.tools.base_v3_obb_hybrid_fixed_test_eval_v51 import (
    CONTRACT_PROTOCOL as V51_EVAL_CONTRACT_PROTOCOL,
    validate_contract as validate_v51_eval_contract)
from crane_project.tools.base_v3_obb_paper_final_report_v1 import (
    METHODS, PROTOCOL as PAPER_REPORT_PROTOCOL, _groups, _riou_metrics)
from crane_project.tools.base_v3_obb_reliability_baseline import _identity, _write_exact
from crane_project.tools.base_v3_obb_true_online_pipeline_v1 import (
    PROTOCOL as TRUE_ONLINE_PROTOCOL)
from crane_project.tools.eval_crane_offline import compute_riou

PROTOCOL = 'base_v3_obb_true_online_finalization_v2'
CONTRACT_PROTOCOL = 'base_v3_obb_true_online_finalization_contract_v2'


def _require_identity(identity, expected, role):
    if identity['sha256'] != expected:
        raise RuntimeError('{} identity mismatch: expected {}, got {}'.format(
            role, expected, identity['sha256']))


def validate_contract(contract):
    if contract.get('protocol') != CONTRACT_PROTOCOL:
        raise ValueError('Unexpected true-online finalization contract')
    for field, expected in {
            'fixed_test_read': True, 'fixed_test_previously_exposed': True,
            'threshold_fitting_performed': False,
            'parameter_tuning_authorized': False}.items():
        if contract.get(field) is not expected:
            raise ValueError('Finalization contract mismatch: '+field)
    expected = contract.get('expected_inputs', {})
    protocols = {
        'true_online_protocol': TRUE_ONLINE_PROTOCOL,
        'paper_report_protocol': PAPER_REPORT_PROTOCOL,
        'v51_eval_contract_protocol': V51_EVAL_CONTRACT_PROTOCOL,
        'attribution_protocol': ATTRIBUTION_PROTOCOL}
    for field, value in protocols.items():
        if expected.get(field) != value:
            raise ValueError('Contract input protocol changed: '+field)
    if contract.get('methods') != list(METHODS):
        raise ValueError('Finalization method inventory changed')
    if contract.get('components') != list(COMPONENTS):
        raise ValueError('Finalization component inventory changed')
    for field in ('center_px', 'side_px', 'angle_deg'):
        if float(contract.get('comparison_tolerances', {}).get(field, -1)) < 0:
            raise ValueError('Invalid comparison tolerance: '+field)


def validate_reports(online, paper, contract):
    scope = contract['fixed_test_scope']
    if online.get('protocol') != TRUE_ONLINE_PROTOCOL:
        raise ValueError('Unexpected true-online report')
    if online.get('execution_mode') != 'true_online_model_inference':
        raise ValueError('Report is not true-online model inference')
    if online.get('pipeline') != contract['pipeline']:
        raise ValueError('True-online pipeline order changed')
    leakage = online.get('leakage_controls', {})
    required = {
        'annotations_or_gt_read': False, 'future_frames_used': False,
        'domain_identity_used_in_model_decisions': False,
        'sequence_frame_identity_used_for_reset_only': True}
    for field, expected in required.items():
        if leakage.get(field) is not expected:
            raise ValueError('True-online leakage control changed: '+field)
    records = online.get('records') or []
    if len(records) != int(scope['frame_count']):
        raise ValueError('True-online frame count changed')
    if len({row['frame_key'] for row in records}) != len(records):
        raise ValueError('True-online report contains duplicate frame keys')
    if dict(Counter(row['domain'] for row in records)) != scope['domain_counts']:
        raise ValueError('True-online domain counts changed')
    if dict(Counter(row['sequence'] for row in records)) != scope['sequence_counts']:
        raise ValueError('True-online sequence counts changed')
    for row in records:
        if 'gt_box' in row or row.get('online_gt_fields_consumed'):
            raise ValueError('True-online record consumed ground truth')
        if set(row.get('observations', {})) != set(METHODS):
            raise ValueError('True-online method inventory is incomplete')
        for method in METHODS:
            if set(row['observations'][method]) != set(COMPONENTS):
                raise ValueError('True-online component inventory is incomplete')
    if paper.get('protocol') != PAPER_REPORT_PROTOCOL:
        raise ValueError('Unexpected historical paper report')
    if len(paper.get('records') or []) != int(scope['frame_count']):
        raise ValueError('Historical paper report frame count changed')
    return records


def _geometry_difference(first, second):
    if first is None or second is None:
        return None
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    fs = sorted((float(first[2]), float(first[3])), reverse=True)
    ss = sorted((float(second[2]), float(second[3])), reverse=True)
    return dict(
        center_px=float(np.linalg.norm(first[:2]-second[:2])),
        long_side_px=abs(fs[0]-ss[0]), short_side_px=abs(fs[1]-ss[1]),
        angle_deg=_periodic_error_deg(float(first[4]), float(second[4])),
        riou=float(compute_riou(first[:5], second[:5])))


def compare_geometry_lane(current, historical, lane, tolerances):
    if len(current) != len(historical):
        raise RuntimeError(
            'Current and historical geometry lanes have different lengths')
    mismatches=[]
    maxima=dict(center_px=0.0, long_side_px=0.0,
                short_side_px=0.0, angle_deg=0.0)
    presence=0
    for a,b in zip(current,historical):
        if a['frame_key'] != b['frame_key']:
            raise RuntimeError('Current and historical frame ordering differs')
        first,second=a.get(lane),b.get(lane)
        presence_changed=(first is None)!=(second is None)
        difference=_geometry_difference(first,second)
        changed=presence_changed
        if difference is not None:
            for field in maxima:
                maxima[field]=max(maxima[field],difference[field])
            changed=bool(
                difference['center_px']>float(tolerances['center_px']) or
                max(difference['long_side_px'],difference['short_side_px'])>
                float(tolerances['side_px']) or
                difference['angle_deg']>float(tolerances['angle_deg']))
        presence += int(presence_changed)
        if changed:
            mismatches.append(dict(
                frame_key=a['frame_key'], current_present=first is not None,
                historical_present=second is not None, difference=difference))
    return dict(
        lane=lane, frame_count=len(current),
        current_present_count=sum(r.get(lane) is not None for r in current),
        historical_present_count=sum(r.get(lane) is not None for r in historical),
        presence_mismatch_count=presence, geometry_mismatch_count=len(mismatches),
        maximum_difference=maxima,
        mismatch_frame_keys=[r['frame_key'] for r in mismatches],
        mismatch_records=mismatches)


def validate_evaluation_alignment(contract, v51_contract):
    """Prevent silent metric-threshold drift between V2 and frozen V5.1."""
    for field in ('center_error_threshold_px',
                  'scale_relative_error_threshold',
                  'angle_error_threshold_deg'):
        if float(contract['evaluation'][field]) != float(
                v51_contract['evaluation'][field]):
            raise RuntimeError('Evaluation threshold changed: ' + field)


def _current_rows(online_records, historical_rows):
    historical={row['frame_key']:row for row in historical_rows}
    rows=[]
    for record in online_records:
        key=record['frame_key']
        if key not in historical:
            raise RuntimeError('True-online frame absent from annotations: '+key)
        detector=record['detector_components']
        rows.append(dict(
            frame_key=key, domain=record['domain'], sequence=record['sequence'],
            frame=int(record['frame']), anchor_source=detector['anchor_source'],
            anchor_score=detector['anchor_score'],
            base_v3_box=copy.deepcopy(detector['base_v3_box']),
            k1_box=copy.deepcopy(detector['k1_box']),
            dino_box=copy.deepcopy(detector['dino_box']),
            gt_box=copy.deepcopy(historical[key]['gt_box'])))
    return rows


def _outputs(records):
    return {method:{row['frame_key']:{
        'components':copy.deepcopy(row['observations'][method])}
        for row in records} for method in METHODS}


def _component_reports(rows, outputs, evaluation):
    return {method:{group:state_report(
        selected,outputs[method],{'evaluation':evaluation},annotated=True)
        for group,selected in _groups(rows)} for method in METHODS}


def _diagnostic_records(rows, online_records):
    online={r['frame_key']:r for r in online_records}; records=[]
    for row in rows:
        observations=copy.deepcopy(online[row['frame_key']]['observations'])
        errors={method:{component:(None if observations[method][component][
            'value'] is None else _error_from_value(
                observations[method][component]['value'],row,component))
            for component in COMPONENTS} for method in METHODS}
        records.append(dict(
            frame_key=row['frame_key'],domain=row['domain'],
            sequence=row['sequence'],frame=int(row['frame']),
            anchor_source=row['anchor_source'],
            online_observations=observations,offline_errors=errors))
    return records


def _delta(a,b):
    return None if a is None or b is None else float(a)-float(b)


def _metric_deltas(component_reports,riou_reports,paper):
    old={(r['method'],r['group'],r['component']):r
         for r in paper['component_metric_table']}
    keys=('measurement_coverage','output_coverage','mean_available_output_error',
          'bad_available_output_rate','correct_output_coverage',
          'longest_unavailable_run')
    components=[]
    for method in METHODS:
        for group,report in component_reports[method].items():
            for component in COMPONENTS:
                current=report[component]; historical=old[(method,group,component)]
                components.append(dict(
                    method=method,group=group,component=component,
                    deltas={key:_delta(current.get(key),historical.get(key))
                            for key in keys}))
    rkeys=('available_obb_coverage','mean_available_riou',
           'riou_hit_coverage','bad_available_obb_rate')
    riou={m:{g:{k:_delta(v.get(k),paper['obb_riou_metrics'][m][g].get(k))
                    for k in rkeys} for g,v in reports.items()}
          for m,reports in riou_reports.items()}
    return dict(component_metric_deltas=components,riou_metric_deltas=riou)


def _csv_rows(records):
    for row in records:
        detector=row['detector_components']
        base=dict(
            frame_key=row['frame_key'],domain=row['domain'],
            sequence=row['sequence'],frame=int(row['frame']),
            timestamp_seconds=row.get('timestamp_seconds'),
            image_width=row['image_size'][0],image_height=row['image_size'][1],
            anchor_source=detector['anchor_source'],
            anchor_score=detector['anchor_score'],
            base_v3_obb=json.dumps(detector['base_v3_box'],separators=(',',':')))
        for method in METHODS:
            for component in COMPONENTS:
                obs=row['observations'][method][component]
                item=dict(base,method=method,component=component)
                item.update(
                    value=json.dumps(obs.get('value'),separators=(',',':')),
                    valid=bool(obs['valid']),state=obs['state'],
                    source=obs.get('source'),
                    age_since_measurement_frames=obs.get(
                        'age_since_measurement_frames'),risk=obs.get('risk'),
                    gate=obs.get('gate'),reason=obs.get('reason'))
                yield item


def write_observation_csv(path,records):
    absolute=Path(path).resolve(); absolute.parent.mkdir(parents=True,exist_ok=True)
    rows=list(_csv_rows(records)); fields=list(rows[0])
    with absolute.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return _identity(absolute)


def run(online,paper,historical_rows,contract,inputs,wall_clock_seconds=None):
    online_records=validate_reports(online,paper,contract)
    current=_current_rows(online_records,historical_rows)
    historical=sorted(historical_rows,key=lambda r:(
        r['domain'],r['sequence'],int(r['frame']),r['frame_key']))
    lane_audit={lane:compare_geometry_lane(
        current,historical,lane,contract['comparison_tolerances'])
        for lane in ('dino_box','k1_box','base_v3_box')}
    outputs=_outputs(online_records); evaluation=contract['evaluation']
    components=_component_reports(current,outputs,evaluation)
    merged={key:{'observations':{m:outputs[m][key]['components'] for m in METHODS}}
            for key in (r['frame_key'] for r in current)}
    riou,riou_by_key=_riou_metrics(
        current,merged,float(evaluation['riou_hit_threshold']))
    diagnostics=_diagnostic_records(current,online_records)
    joint=diagnostic.joint_availability(diagnostics,{'evaluation':evaluation})
    runtime=None
    if wall_clock_seconds is not None:
        seconds=float(wall_clock_seconds)
        if not math.isfinite(seconds) or seconds<=0:
            raise ValueError('wall-clock seconds must be positive and finite')
        runtime=dict(
            scope='full_command_model_init_detector_image_io_and_reporting',
            wall_clock_seconds=seconds,average_frames_per_second=len(current)/seconds,
            average_seconds_per_frame=seconds/len(current),multi_gpu_parallelism=False)
    return dict(
        protocol=PROTOCOL,evidence_boundary=contract['evidence_boundary'],
        claim_status='POST_EXPOSURE_TRUE_ONLINE_FINALIZATION_V2',inputs=inputs,
        fixed_test_read=True,fixed_test_previously_exposed=True,
        threshold_fitting_performed=False,feature_selection_performed=False,
        policy_selection_performed=False,parameter_tuning_authorized=False,
        leakage_controls=dict(
            gt_used_in_model_inference=False,gt_used_in_online_observation=False,
            gt_attached_after_true_online_output=True,
            gt_used_only_for_offline_metrics=True,future_frames_used=False,
            domain_identity_used_in_model_decisions=False),
        dataset_summary=copy.deepcopy(contract['fixed_test_scope']),
        evaluation_thresholds=copy.deepcopy(evaluation),
        candidate_and_geometry_reproduction=lane_audit,
        component_metrics=components,joint_component_metrics=joint,
        obb_riou_metrics=riou,
        comparison_with_cached_paper_report=_metric_deltas(components,riou,paper),
        runtime_measurement=runtime,
        records=[dict(frame_key=row['frame_key'],
                      offline_errors=diagnostics[i]['offline_errors'],
                      offline_riou=riou_by_key[row['frame_key']])
                 for i,row in enumerate(current)],
        online_observation_export_contains_gt=False,
        eligible_for_parameter_tuning_from_this_report=False,
        eligible_for_fresh_unknown_sequence_claim=False,
        claim_limit=contract['claim_limit'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--true-online-report',required=True)
    parser.add_argument('--paper-final-report',required=True)
    parser.add_argument('--fixed-test-attribution',required=True)
    parser.add_argument('--v51-eval-contract',required=True)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--base-v3-results'); parser.add_argument('--k1-results')
    parser.add_argument('--all-lane-audit'); parser.add_argument('--ann-dir')
    parser.add_argument('--wall-clock-seconds',type=float)
    parser.add_argument('--out-observations-csv',required=True)
    parser.add_argument('--out-json',required=True)
    args=parser.parse_args()
    paths=dict(true_online_report=args.true_online_report,
               paper_final_report=args.paper_final_report,
               fixed_test_attribution=args.fixed_test_attribution,
               v51_eval_contract=args.v51_eval_contract,contract=args.contract)
    identities={role:_identity(path) for role,path in paths.items()}
    contract=json.loads(Path(args.contract).read_text(encoding='utf-8'))
    validate_contract(contract); expected=contract['expected_inputs']
    for role,field in (
            ('true_online_report','true_online_sha256'),
            ('paper_final_report','paper_report_sha256'),
            ('fixed_test_attribution','attribution_sha256'),
            ('v51_eval_contract','v51_eval_contract_sha256')):
        _require_identity(identities[role],expected[field],role)
    online=json.loads(Path(args.true_online_report).read_text(encoding='utf-8'))
    paper=json.loads(Path(args.paper_final_report).read_text(encoding='utf-8'))
    attribution=json.loads(Path(args.fixed_test_attribution).read_text(encoding='utf-8'))
    v51=json.loads(Path(args.v51_eval_contract).read_text(encoding='utf-8'))
    validate_v51_eval_contract(v51)
    validate_evaluation_alignment(contract, v51)
    historical,reconstruction=build_records(attribution,args,v51)
    identities['historical_reconstruction']=reconstruction
    payload=run(online,paper,historical,contract,identities,args.wall_clock_seconds)
    csv_identity=write_observation_csv(args.out_observations_csv,online['records'])
    payload['online_observation_export']=csv_identity
    output=_write_exact(args.out_json,payload)
    lane=payload['candidate_and_geometry_reproduction']
    print(json.dumps(dict(
        output=output,online_observation_export=csv_identity,
        frame_count=payload['dataset_summary']['frame_count'],
        candidate_and_geometry_reproduction={name:{key:value[key] for key in (
            'current_present_count','historical_present_count',
            'presence_mismatch_count','geometry_mismatch_count',
            'maximum_difference','mismatch_frame_keys')}
            for name,value in lane.items()},
        joint_all_summary={method:{key:payload['joint_component_metrics'][method][
            'all'][key] for key in ('all_components_valid_coverage',
            'jointly_correct_coverage','false_all_components_valid_rate',
            'longest_incomplete_obb_run')} for method in METHODS},
        riou_all_summary={method:{key:payload['obb_riou_metrics'][method]['all'][key]
            for key in ('available_obb_coverage','mean_available_riou',
            'riou_hit_coverage','bad_available_obb_rate')} for method in METHODS},
        runtime_measurement=payload['runtime_measurement'],
        claim_status=payload['claim_status'],parameter_tuning_authorized=False),
        indent=2))


if __name__=='__main__':
    main()
