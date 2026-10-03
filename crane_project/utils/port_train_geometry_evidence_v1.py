"""CPU evidence utilities. No detector, feature cache, optimizer or relabeling."""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from crane_project.tools.audit_port_train_val_geometry_v1 import canonical, describe

SETTINGS = dict(
    center_bins_px=[1.,2.,5.,15.], short_cell_bins=[2.,4.], aspect_bins=[1.2,2.],
    center_tolerance_px=1e-6, riou_tolerance=1e-5,
    raw_crop_context=2.5, max_display_zoom=6., tile_size=420,
    crop_height=360, image_limit=12, native_stride=8)
COLORS = dict(annotation=(255,255,255),gt=(90,240,110),b=(255,180,40),
              joint=(80,210,255),center_only=(255,100,220),lattice=(90,120,160))


def box_polygon(box):
    b = np.asarray(box,dtype=float)
    if b.shape not in ((5,),(6,)) or not np.isfinite(b).all() or min(b[2:4])<=0:
        raise ValueError('Expected finite positive original OBB')
    c,s = math.cos(b[4]),math.sin(b[4])
    a=np.array([c,s])*b[2]/2.; d=np.array([-s,c])*b[3]/2.
    return np.stack([b[:2]-a-d,b[:2]+a-d,b[:2]+a+d,b[:2]-a+d])


def read_annotation(path):
    """Use the actual CraneDataset poly2obb_np le90 convention, retain raw GT."""
    from mmrotate.core import poly2obb_np
    objects=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=10 or fields[8]!='grab':
            raise ValueError('Expected one strict TRAIN grab DOTA object: '+str(path))
        poly=np.asarray([float(v) for v in fields[:8]],dtype=np.float32).reshape(4,2)
        if not np.isfinite(poly).all() or not cv2.isContourConvex(poly):
            raise ValueError('Nonfinite/nonconvex TRAIN polygon: '+str(path))
        parsed=poly2obb_np(poly.reshape(-1),version='le90')
        if parsed is None: raise ValueError('Unparseable TRAIN polygon')
        objects.append(dict(polygon=poly.astype(float).tolist(),parsed_obb=list(parsed),
                            category=fields[8],difficulty=int(fields[9])))
    if len(objects)!=1: raise ValueError('Expected exactly one TRAIN annotation')
    return objects[0]


def annotation_evidence(sample,annotation):
    p=np.asarray(annotation['polygon']); parsed=canonical(annotation['parsed_obb']); gt=canonical(sample['gt'])
    matched=bool(np.allclose(parsed,gt,atol=1e-3,rtol=1e-5))
    rect=box_polygon(parsed); area=abs(float(cv2.contourArea(p.astype(np.float32))))
    w,h=sample['image_size']
    margin=min(float(p[:,0].min()),float(p[:,1].min()),w-1-float(p[:,0].max()),h-1-float(p[:,1].max()))
    # Corner order is arbitrary; minimize over cyclic order AND winding.
    distances=[np.linalg.norm(p-np.roll(q,k,axis=0),axis=1) for q in (rect,rect[::-1]) for k in range(4)]
    best=min(distances,key=lambda d:float(d.mean()))
    return dict(image=sample['image'],raw_polygon=annotation['polygon'],parsed_obb=parsed.tolist(),
        saved_reference_obb=gt.tolist(),parsed_minus_saved= (parsed-gt).tolist(),
        native_parse_reference_status='WITHIN_ORIGINAL_REFERENCE_TOLERANCE' if matched else 'NATIVE_PARSE_DIFFERENCE_REVIEW_REQUIRED',
        native_parse_center_difference_px=float(np.linalg.norm(parsed[:2]-gt[:2])),
        native_parse_periodic_angle_difference_deg=float(abs((parsed[4]-gt[4]+math.pi/2)%math.pi-math.pi/2)*180/math.pi),
        polygon_area_px2=area,rectangle_area_px2=float(parsed[2]*parsed[3]),
        polygon_rectangle_area_ratio=area/(parsed[2]*parsed[3]),
        polygon_to_fitted_rectangle_corner_mean_px=float(best.mean()),
        polygon_to_fitted_rectangle_corner_max_px=float(best.max()),
        image_edge_margin_px=margin,image_edge_margin_over_gt_short=margin/gt[3],
        vertices_inside_image=margin>=0,difficulty=annotation['difficulty'],
        semantic_label_correctness='NOT_REVIEWED',visibility='NOT_REVIEWED',
        note='Bytes are checked upstream. Preserve fixed saved GT for metrics; native minAreaRect can differ numerically by build. Geometric consistency does not establish semantic label quality.')


def view_geometry(image_size,scale):
    """Exact native resize dimensions; isotropic shrink after rounded RResize."""
    import mmcv
    if scale not in (1.,.5): raise ValueError('Undeclared TRAIN view scale')
    w,h=image_size
    (nw,nh),_ = mmcv.rescale_size((w,h),(1024,1024),return_scale=True)
    factors=np.asarray([nw/w,nh/h,nw/w,nh/h],dtype=np.float32)*scale
    return dict(scale=scale,resize_wh=[nw,nh],
        image_wh=[int(math.ceil(nw*scale)),int(math.ceil(nh*scale))],
        scale_factor=factors.astype(float).tolist(),pad_wh=[1024,1024],
        flip=False,coordinate_convention='top-left; detector raw sizes*sx/sy; annotation sizes*sqrt(sx*sy)',
        image_size_rounding_only=True)


def model_box(box,geometry,annotation=False):
    b=np.asarray(box,dtype=float).copy()
    sx,sy=geometry['scale_factor'][:2]
    b[:2]*=[sx,sy]
    b[2:4]*=math.sqrt(sx*sy) if annotation else np.array([sx,sy])
    return b


def native_view(rgb,scale):
    """Native CPU pixel transforms, no Normalize/Pad/dataset construction."""
    from mmrotate.datasets.pipelines.transforms import RResize
    from mmrotate.datasets.pipelines.port_train_augment import PortIsotropicShrink
    resized=RResize(img_scale=(1024,1024))(dict(
        img=np.asarray(rgb).copy(),img_fields=['img'],bbox_fields=[],ori_shape=np.asarray(rgb).shape))
    if scale==.5:
        resized=PortIsotropicShrink(prob=1.,scale_range=(.5,.5))(resized)
    elif scale!=1.: raise ValueError('Undeclared scale')
    geometry=view_geometry(list(rgb.size),scale)
    if (list(resized['img'].shape[1::-1])!=geometry['image_wh'] or
            np.asarray(resized['scale_factor']).tolist()!=geometry['scale_factor']):
        raise ValueError('Native pixel transform geometry differs')
    return Image.fromarray(resized['img']),geometry


def native_rgb(path):
    """Use the actual view loader, not an independent Pillow JPEG decoder."""
    from mmdet.datasets.pipelines import LoadImageFromFile
    value=LoadImageFromFile()(dict(img_prefix=None,img_info=dict(filename=str(path))))
    pixels=value['img']
    if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:
        raise ValueError('Expected native uint8 BGR image')
    # Resize/shrink act independently per channel. This permutation preserves
    # the native decoded pixel values while making display RGB explicit.
    return Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))


def bin_name(value,boundaries):
    for b in boundaries:
        if value<b: return 'lt_'+str(b)
    return 'ge_'+str(boundaries[-1])


def center_parts(pred,gt,b):
    if pred is None or b is None: return None
    e=(np.asarray(pred)[:2]-np.asarray(gt)[:2])/(.3*min(b[2:4]))
    parts=np.where(abs(e)<.1,.5*e*e/.1,abs(e)-.05)
    return dict(raw=parts.tolist(),weighted_sum=float(parts.sum()/3.),
                loss_convention='(px+py)/3; original frozen B short edge')


def build_rows(arms,samples,annotations):
    lookup={s['image']:s for s in samples}; rows=[]
    reference=arms['b']['rows']
    for i,br in enumerate(reference):
        sample=lookup[br['image']]; gt=canonical(br['gt_original']); b=br['pred']
        geom=view_geometry(sample['image_size'],br['scale'])
        models=model_box(gt,geom,annotation=True)
        item={k:br[k] for k in ('image','domain','sequence','frame_id','role','scale','eligible')}
        item.update(gt_original=br['gt_original'],geometry=geom,
            annotation=annotations[br['image']],gt_aspect=float(gt[2]/gt[3]),
            gt_input_short_cells=float(models[3]/SETTINGS['native_stride']),
            gt_short_cells_bin=bin_name(models[3]/SETTINGS['native_stride'],SETTINGS['short_cell_bins']),
            aspect_bin=bin_name(gt[2]/gt[3],SETTINGS['aspect_bins']),
            baseline_center_bin=bin_name(br['metrics']['center_error_px'],SETTINGS['center_bins_px']) if b else 'no_output',
            arms={})
        if b:
            desired=gt[:2]-np.asarray(b)[:2]; short=min(b[2:4]); bc=model_box(b,geom)[:2]/SETTINGS['native_stride']
            item.update(desired_shift_px=desired.tolist(),desired_shift_over_b_short=(desired/short).tolist(),
                desired_shift_model_px=(desired*np.asarray(geom['scale_factor'][:2])).tolist(),
                b_short_original_px=float(short),b_center_stride8_phase=(bc-np.floor(bc)).tolist(),
                phase_note='Anchor-coordinate stride8 lattice only; not measured FPN receptive field or response.')
        for label,arm in arms.items():
            cr=arm['rows'][i]; pred=cr['pred']; m=cr['metrics']
            record=dict(pred=pred,metrics=m,center_loss=center_parts(pred,br['gt_original'],b))
            if b and pred:
                actual=np.asarray(pred)[:2]-np.asarray(b)[:2]; des=np.asarray(item['desired_shift_px'])
                den=float(np.linalg.norm(actual)*np.linalg.norm(des))
                record.update(actual_shift_px=actual.tolist(),actual_shift_over_b_short=(actual/short).tolist(),
                    signed_residual_px=(np.asarray(pred)[:2]-gt[:2]).tolist(),
                    signed_residual_over_b_short=((np.asarray(pred)[:2]-gt[:2])/short).tolist(),
                    direction_cosine=float(np.dot(actual,des)/den) if den>1e-12 else None,
                    center_delta_px=m['center_error_px']-br['metrics']['center_error_px'],
                    riou_delta=m['riou']-br['metrics']['riou'])
            item['arms'][label]=record
        rows.append(item)
    return rows


def grouped_report(rows):
    """Keep roles and scales separate in every stratum; no pseudo-replication."""
    groups={}
    for row in rows:
        base=row['role']+'/'+row['domain']+'/'+str(row['scale'])
        keys=[base+'/all',base+'/sequence/'+row['sequence'],
              base+'/baseline_center/'+row['baseline_center_bin'],base+'/short_cells/'+row['gt_short_cells_bin'],
              base+'/aspect/'+row['aspect_bin']]
        for key in keys: groups.setdefault(key,[]).append(row)
    results={}
    for key,group in groups.items():
        summaries={}
        for label in group[0]['arms']:
            rec=[r['arms'][label] for r in group]; outs=[r for r in rec if r['metrics']['output']]
            hit=sum(r['metrics']['center_hit'] for r in outs); riou=[r['metrics']['riou'] for r in rec]
            directions=[r['direction_cosine'] for r in outs if r.get('direction_cosine') is not None]
            summaries[label]=dict(frames=len(group),
                output_coverage=dict(numerator=len(outs),denominator=len(group),pct=100*len(outs)/len(group)),
                conditional_center_correct=dict(numerator=hit,denominator=len(outs),pct=100*hit/len(outs) if outs else None),
                all_frame_center_correct=dict(numerator=hit,denominator=len(group),pct=100*hit/len(group)),
                center_px=describe([r['metrics']['center_error_px'] for r in outs]),
                center_over_gt_short=describe([r['metrics']['center_error_over_gt_short'] for r in outs]),
                center_loss=describe([r['center_loss']['weighted_sum'] for r in outs if r['center_loss'] is not None]),
                riou=describe(riou),riou_p10=float(np.percentile(riou,10)),riou_min=min(riou),
                pure_angle_deg=describe([r['metrics']['angle_error_deg'] for r in outs]),
                protocol_angle_deg_all_frames=describe([r['metrics']['protocol_angle_error_deg'] for r in rec]),
                long_edge_error=describe([r['metrics']['long_edge_relative_error'] for r in outs]),
                short_edge_error=describe([r['metrics']['short_edge_relative_error'] for r in outs]),
                direction_cosine=describe(directions),
                center_better=sum(r.get('center_delta_px',0)<-SETTINGS['center_tolerance_px'] for r in outs),
                center_worse=sum(r.get('center_delta_px',0)>SETTINGS['center_tolerance_px'] for r in outs),
                riou_better=sum(r.get('riou_delta',0)>SETTINGS['riou_tolerance'] for r in outs),
                riou_worse=sum(r.get('riou_delta',0)<-SETTINGS['riou_tolerance'] for r in outs),
                signed_residual_px={axis:describe([r['signed_residual_px'][i] for r in outs if 'signed_residual_px' in r]) for i,axis in enumerate(('x','y'))},
                signed_residual_over_b_short={axis:describe([r['signed_residual_over_b_short'][i] for r in outs if 'signed_residual_over_b_short' in r]) for i,axis in enumerate(('x','y'))},
                actual_shift_px={axis:describe([r['actual_shift_px'][i] for r in outs if 'actual_shift_px' in r]) for i,axis in enumerate(('x','y'))},
                actual_shift_over_b_short={axis:describe([r['actual_shift_over_b_short'][i] for r in outs if 'actual_shift_over_b_short' in r]) for i,axis in enumerate(('x','y'))},
                actual_shift_sign_counts={axis:dict(negative=sum(r['actual_shift_px'][i]<0 for r in outs if 'actual_shift_px' in r),
                    positive=sum(r['actual_shift_px'][i]>0 for r in outs if 'actual_shift_px' in r),
                    zero=sum(r['actual_shift_px'][i]==0 for r in outs if 'actual_shift_px' in r)) for i,axis in enumerate(('x','y'))})
        results[key]=dict(views=len(group),image_names=[r['image'] for r in group],arms=summaries,
            desired_shift_px={axis:describe([r['desired_shift_px'][i] for r in group if 'desired_shift_px' in r]) for i,axis in enumerate(('x','y'))},
            desired_shift_over_b_short={axis:describe([r['desired_shift_over_b_short'][i] for r in group if 'desired_shift_over_b_short' in r]) for i,axis in enumerate(('x','y'))},
            desired_shift_sign_counts={axis:dict(negative=sum(r['desired_shift_px'][i]<0 for r in group if 'desired_shift_px' in r),
                positive=sum(r['desired_shift_px'][i]>0 for r in group if 'desired_shift_px' in r),
                zero=sum(r['desired_shift_px'][i]==0 for r in group if 'desired_shift_px' in r)) for i,axis in enumerate(('x','y'))},
            gt_short_cells=describe([r['gt_input_short_cells'] for r in group]))
    return results


def select_images(rows):
    """Diagnostic extremes plus median control; no fitting or method selection."""
    slots=[]
    for role in ('fit','probe'):
        for domain in ('real','sim'):
            subset=[r for r in rows if r['role']==role and r['domain']==domain]
            used=set()
            for kind in ('negative','positive','ordinary'):
                available=[r for r in subset if r['image'] not in used and r['arms']['center_only']['metrics']['output']]
                if kind=='negative':
                    candidates=[r for r in available if r['arms']['center_only']['center_delta_px']>SETTINGS['center_tolerance_px'] and r['arms']['center_only']['riou_delta']<-SETTINGS['riou_tolerance']]
                    ordered=sorted(candidates,key=lambda r:(r['arms']['center_only']['riou_delta'],-r['arms']['center_only']['center_delta_px'],r['image'],r['scale']))
                elif kind=='positive':
                    candidates=[r for r in available if r['arms']['center_only']['center_delta_px']<-SETTINGS['center_tolerance_px'] and r['arms']['center_only']['riou_delta']>SETTINGS['riou_tolerance']]
                    ordered=sorted(candidates,key=lambda r:(-r['arms']['center_only']['riou_delta'],r['arms']['center_only']['center_delta_px'],r['image'],r['scale']))
                else:
                    candidates=[r for r in available if r['scale']==1.]
                    median=float(np.median([r['arms']['b']['metrics']['center_error_px'] for r in subset if r['scale']==1. and r['arms']['b']['metrics']['output']]))
                    ordered=sorted(candidates,key=lambda r:(abs(r['arms']['b']['metrics']['center_error_px']-median),r['image']))
                slot=dict(role=role,domain=domain,category=kind,candidate_views=len(ordered),
                    ordered_candidates=[dict(image=r['image'],scale=r['scale']) for r in ordered],
                    status='SELECTED' if ordered else 'UNAVAILABLE_NO_FALLBACK')
                if ordered:
                    row=ordered[0]; used.add(row['image'])
                    slot.update(image=row['image'],trigger_scale=row['scale'],display_scales=[1.,.5],
                        center_delta_px=row['arms']['center_only']['center_delta_px'],riou_delta=row['arms']['center_only']['riou_delta'])
                slots.append(slot)
    selected=[s['image'] for s in slots if s['status']=='SELECTED']
    if len(selected)>SETTINGS['image_limit'] or len(selected)!=len(set(selected)):
        raise ValueError('Diagnostic selection budget or uniqueness differs')
    return dict(slots=slots,unique_images=len(selected),
        note='Result-guided TRAIN diagnosis, not random validation or a performance benchmark; all128 views still reported. No cherry-picked method selection.')


def crop_bounds(points,image_size):
    p=np.concatenate([np.asarray(x,dtype=float).reshape(-1,2) for x in points])
    low,high=p.min(axis=0),p.max(axis=0); center=(low+high)/2.
    extent=np.maximum(high-low,16.)*SETTINGS['raw_crop_context']
    w,h=image_size
    x0,y0=np.maximum(np.floor(center-extent/2.),0).astype(int)
    x1,y1=np.minimum(np.ceil(center+extent/2.),[w,h]).astype(int)
    if x1<=x0 or y1<=y0: raise ValueError('Empty evidence crop')
    return [int(x0),int(y0),int(x1),int(y1)]


def display_transform(bounds):
    x0,y0,x1,y1=bounds; side=SETTINGS['tile_size']; height=SETTINGS['crop_height']
    zoom=min(side/(x1-x0),height/(y1-y0),SETTINGS['max_display_zoom'])
    return dict(crop_xyxy=bounds,display_zoom=zoom,
        paste_xy=[(side-(x1-x0)*zoom)/2.,(height-(y1-y0)*zoom)/2.],
        interpolation='nearest; display only, no additional spatial evidence',uniform_scale=True)


def display_points(points,transform):
    return ((np.asarray(points,dtype=float)-np.asarray(transform['crop_xyxy'][:2]))*
            transform['display_zoom']+np.asarray(transform['paste_xy']))


def crop_tile(image,bounds,polygons=None,centers=None,lattice=False):
    transform=display_transform(bounds); zoom=transform['display_zoom']
    # One affine map for pixels AND overlays; no independent integer resizing
    # of width/height or rounded paste position at subpixel correction scales.
    crop=np.asarray(image.crop(tuple(bounds)))
    matrix=np.array([[zoom,0,transform['paste_xy'][0]],
                     [0,zoom,transform['paste_xy'][1]]],dtype=np.float64)
    tile=Image.fromarray(cv2.warpAffine(crop,matrix,
        (SETTINGS['tile_size'],SETTINGS['crop_height']),flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,borderValue=(18,22,28)))
    draw=ImageDraw.Draw(tile)
    if lattice:
        x0,y0,x1,y1=bounds
        for y in range(int(math.ceil(y0/8))*8,y1,8):
            for x in range(int(math.ceil(x0/8))*8,x1,8):
                u,v=display_points([[x,y]],transform)[0]; draw.ellipse((u-1,v-1,u+1,v+1),fill=COLORS['lattice'])
    for name,points in (polygons or {}).items():
        p=display_points(points,transform).tolist(); draw.line([tuple(x) for x in p+[p[0]]],fill=COLORS[name],width=2)
    for name,point in (centers or {}).items():
        u,v=display_points([point],transform)[0]
        draw.line((u-4,v,u+4,v),fill=COLORS[name],width=2); draw.line((u,v-4,u,v+4),fill=COLORS[name],width=2)
    return tile,transform


def render_panel(sample,annotation,paired_rows,path):
    """Raw and overlay for each of three views, plus whole-frame context."""
    rgb=native_rgb(sample['path'])
    if list(rgb.size)!=sample['image_size']: raise ValueError('Image geometry changed')
    rows={r['scale']:r for r in paired_rows}
    if set(rows)!={1.,.5}: raise ValueError('Expected both correlated views')
    polygon=np.asarray(annotation['polygon']); gt=rows[1.]['gt_original']
    footprints=[polygon,box_polygon(gt)]
    for row in paired_rows:
        footprints += [box_polygon(row['arms'][k]['pred']) for k in ('b','joint','center_only') if row['arms'][k]['pred']]
    bounds=crop_bounds(footprints,rgb.size)
    polygons=dict(annotation=polygon,gt=box_polygon(gt))
    centers=dict(gt=gt[:2])
    for label in ('b','joint','center_only'):
        pred=rows[1.]['arms'][label]['pred']
        if pred: polygons[label]=box_polygon(pred); centers[label]=pred[:2]
    raw,rawmap=crop_tile(rgb,bounds)
    overlay,overlaymap=crop_tile(rgb,bounds,polygons,centers)
    raw_tiles=[raw]; overlays=[overlay]; maps=dict(original_raw=rawmap,original_overlay=overlaymap)
    for scale in (1.,.5):
        image,geom=native_view(rgb,scale); row=rows[scale]
        # Same original crop extent mapped into this real pixel view.
        sx,sy=geom['scale_factor'][:2]
        bb=[int(math.floor(bounds[0]*sx)),int(math.floor(bounds[1]*sy)),
            min(image.width,int(math.ceil(bounds[2]*sx))),min(image.height,int(math.ceil(bounds[3]*sy)))]
        pp=dict(annotation=polygon*np.array([sx,sy]),gt=box_polygon(model_box(row['gt_original'],geom,annotation=True)))
        cc=dict(gt=model_box(row['gt_original'],geom,annotation=True)[:2])
        for label in ('b','joint','center_only'):
            pred=row['arms'][label]['pred']
            if pred:
                pb=model_box(pred,geom); pp[label]=box_polygon(pb); cc[label]=pb[:2]
        native_raw,mapping=crop_tile(image,bb)
        tile,_=crop_tile(image,bb,pp,cc,lattice=True)
        raw_tiles.append(native_raw); overlays.append(tile)
        maps[str(scale)]=dict(geometry=geom,display=mapping,
            note='Reconstructed native input pixels and anchor-coordinate stride8 dots; not FPN activations or learned ROI response.')
    side=SETTINGS['tile_size']; panel=Image.new('RGB',(side*3,1060),(18,22,28)); draw=ImageDraw.Draw(panel)
    draw.text((12,10),sample['image']+'  '+sample['role']+' / '+sample['domain'],fill=(240,240,240))
    labels=['Original pixels','Native input scale1.0','Native input scale0.5']
    for i,label in enumerate(labels):
        panel.paste(raw_tiles[i],(i*side,58)); panel.paste(overlays[i],(i*side,450))
        draw.text((i*side+8,38),label+' : raw',fill=(220,220,220))
        draw.text((i*side+8,430),label+' : overlay',fill=(220,220,220))
        if i:
            row=rows[1. if i==1 else .5]; geometry=row['geometry']
            info='Input %dx%d | GT short %.2fpx / %.2f stride8 cells'%(
                *geometry['image_wh'],row['gt_input_short_cells']*8,row['gt_input_short_cells'])
            draw.text((i*side+8,16),info,fill=(200,200,200))
    y=824
    for name in ('annotation','gt','b','joint','center_only'):
        draw.text((12+list(COLORS).index(name)*180,y),name,fill=COLORS[name])
    for j,scale in enumerate((1.,.5)):
        row=rows[scale]; text='scale '+str(scale)+' | '
        text+=' | '.join(k+': center %.3fpx, RIoU %.6f'%(row['arms'][k]['metrics']['center_error_px'],row['arms'][k]['metrics']['riou'])
                         for k in ('b','joint','center_only'))
        draw.text((12,850+j*24),text,fill=(230,230,230))
    draw.text((12,907),'White: raw DOTA polygon; green: saved GT OBB. Nearest display enlargement; no sharpening.',fill=(210,210,210))
    draw.text((12,929),'Blue dots are coordinate lattice only, not FPN activations. Raw images above remain unmarked.',fill=(210,210,210))
    thumb=rgb.copy(); thumb.thumbnail((180,80),Image.BILINEAR); panel.paste(thumb,(12,960))
    # Context rectangle uses uniform thumbnail scale with native rounding.
    td=ImageDraw.Draw(panel); tx,ty=thumb.width/rgb.width,thumb.height/rgb.height
    td.rectangle((12+bounds[0]*tx,960+bounds[1]*ty,12+bounds[2]*tx,960+bounds[3]*ty),outline=(255,255,255),width=1)
    draw.text((210,974),'Whole-frame thumbnail only. Label/visibility/boundary evidence require human review.',fill=(200,200,200))
    draw.text((210,996),'No relabeling, detector updates, feature measurement or depth/continuity validation.',fill=(200,200,200))
    panel.save(path)
    return dict(image=sample['image'],file=path.name,panel_wh=list(panel.size),crop_maps=maps,
                pixel_render_backend='native LoadImageFromFile BGR->RGB + RResize + PortIsotropicShrink; uniform cv2 nearest display affine')
