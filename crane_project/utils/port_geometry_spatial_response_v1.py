"""Read-only P3/ROI measurements; no head, optimizer, labels or output policy."""
import numpy as np
from PIL import Image, ImageDraw
import torch

from crane_project.utils import port_geometry_refine_g_v1 as local


SETTINGS = dict(image_limit=12, scales=[1., .5], view_limit=24,
    stride=8, channels=256, roi_size=9, shift_model_px=1.,
    patch_margin_cells=1, max_patch_side_cells=64,
    agreement_atol=1e-5, agreement_rtol=1e-4, ratio_epsilon=1e-12)
SHIFTS = dict(x_minus=(-1., 0.), x_plus=(1., 0.),
              y_minus=(0., -1.), y_plus=(0., 1.))


def checked_feature(value):
    if (value.ndim != 4 or value.shape[:2] != (1, SETTINGS['channels'])
            or value.requires_grad or value.grad_fn is not None
            or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
        raise ValueError('Expected finite detached batch1 float32 256-channel features')
    if min(value.shape[-2:]) < 2:
        raise ValueError('Spatial field too small')


def rms(value):
    # Double accumulation prevents squaring float32 features from overflowing.
    return float(value.double().square().mean().sqrt())


def ratio(value, base):
    return value/base if base > SETTINGS['ratio_epsilon'] else None


def summarize_feature(value):
    checked_feature(value)
    x = value.double()[0]
    centered = x-x.mean(dim=(1, 2), keepdim=True)
    total, spatial = rms(x), rms(centered)
    per_channel = centered.square().mean(dim=(1, 2)).sqrt().cpu().numpy()
    return dict(total_rms=total, spatial_demeaned_rms=spatial,
        spatial_over_total=ratio(spatial, total),
        x_neighbor_difference_rms=rms(x[:, :, 1:]-x[:, :, :-1]),
        y_neighbor_difference_rms=rms(x[:, 1:, :]-x[:, :-1, :]),
        channel_spatial_rms_quantiles=np.quantile(per_channel, [0., .5, .9, 1.]).tolist(),
        maps=dict(channel_rms=x.square().mean(dim=0).sqrt().cpu().tolist(),
                  channel_spatial_deviation_rms=centered.square().mean(dim=0).sqrt().cpu().tolist()),
        note='All256 channels; energy/variation, not an object probability or localization score.')


def agreement(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Re-extracted/cache tensor shape or dtype differs')
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError('Nonfinite tensor agreement')
    delta = actual.double()-expected.to(actual.device).double()
    result = dict(max_absolute_delta=float(delta.abs().max()), rms_delta=rms(delta),
        rms_delta_over_cache=ratio(rms(delta), rms(expected)),
        passed=bool(torch.allclose(actual, expected.to(actual.device),
            atol=SETTINGS['agreement_atol'], rtol=SETTINGS['agreement_rtol'])))
    if not result['passed']:
        raise ValueError('Re-extracted ROI/support differs from reviewed cache: '+str(result))
    return result


def roi_coordinates(points, box, mode):
    """Fractional grid indices, endpoints 0/8; not a new feature resolution."""
    b = local.canonical_boxes(box.detach())[0]
    side = (b[2:4]*local.SETTINGS['context_multiplier']).clamp(
        min=SETTINGS['stride']*local.SETTINGS['min_context_cells'])
    angle = b[4] if mode == 'aligned' else b.new_tensor(0.)
    d = points.to(b)-b[:2]
    unrotated = torch.stack((d[..., 0]*angle.cos()+d[..., 1]*angle.sin(),
                            -d[..., 0]*angle.sin()+d[..., 1]*angle.cos()), dim=-1)
    return ((unrotated/side+.5)*(SETTINGS['roi_size']-1)).cpu().tolist()


def shift_responses(p3, boxes_model, meta):
    """Translate sampling center ONLY; never move image, shape, score or GT."""
    checked_feature(p3)
    if boxes_model.shape != (1, 5):
        raise ValueError('Exactly one frozen B box required')
    base, support, points = local.sample_local(p3, boxes_model, meta, 'ordinary')
    features, result = {}, {}
    base_rms = rms(base)
    for name, unit in SHIFTS.items():
        delta = boxes_model.new_tensor(unit)*SETTINGS['shift_model_px']
        shifted = boxes_model.clone(); shifted[:, :2] += delta
        value, mask, _ = local.sample_local(p3, shifted, meta, 'ordinary')
        features[name] = value
        diff = value.double()-base.double()
        magnitude = rms(diff)
        result[name] = dict(displacement_model_px=delta.cpu().tolist(),
            displacement_original_px=(delta/delta.new_tensor(meta['scale_factor'][:2])).cpu().tolist(),
            difference_rms=magnitude, difference_over_base_rms=ratio(magnitude, base_rms),
            difference_map=diff.square().mean(dim=1)[0].sqrt().cpu().tolist(),
            support_min=float(mask.min()), support_mean=float(mask.mean()),
            support_max_absolute_delta=float((mask-support).abs().max()))
    derivative = {}
    for axis in ('x', 'y'):
        d = (features[axis+'_plus'].double()-features[axis+'_minus'].double())/(2*SETTINGS['shift_model_px'])
        derivative[axis] = dict(rms_per_model_px=rms(d),
            rms_per_original_px=rms(d)*float(meta['scale_factor'][0 if axis == 'x' else 1]),
            map_per_model_px=d.square().mean(dim=1)[0].sqrt().cpu().tolist())
    return dict(shifts=result, central_difference=derivative), points


def native_patch(p3, sample_points, reference_centers, meta):
    """Native P3 cell window; retain index origin and padding mask separately."""
    checked_feature(p3)
    points = torch.cat((sample_points.reshape(-1, 2), reference_centers.reshape(-1, 2)))
    margin = SETTINGS['patch_margin_cells']
    low = torch.floor((points.min(dim=0)[0]-SETTINGS['shift_model_px'])/SETTINGS['stride']).long()-margin
    high = torch.ceil((points.max(dim=0)[0]+SETTINGS['shift_model_px'])/SETTINGS['stride']).long()+margin
    h, w = p3.shape[-2:]
    x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
    x1, y1 = min(w-1, int(high[0])), min(h-1, int(high[1]))
    if (x1 <= x0 or y1 <= y0 or
            max(x1-x0+1, y1-y0+1) > SETTINGS['max_patch_side_cells']):
        raise ValueError('Native P3 patch exceeds fixed budget or is empty')
    patch = p3[:, :, y0:y1+1, x0:x1+1]
    stats = summarize_feature(patch)
    ys = torch.arange(y0, y1+1, device=p3.device)*SETTINGS['stride']
    xs = torch.arange(x0, x1+1, device=p3.device)*SETTINGS['stride']
    valid = (ys[:, None] < meta['img_shape'][0]) & (xs[None, :] < meta['img_shape'][1])
    stats.update(index_bounds_inclusive=[x0, y0, x1, y1],
        model_coordinate_rule='(index_x*8,index_y*8); native cell samples, not receptive-field bounds',
        valid_mask=valid.int().cpu().tolist(), feature_shape=list(p3.shape),
        padding_note='Patch energies are raw P3; mask shown separately. ROI uses the reviewed masked sampler.')
    return stats


def heat_tile(values, title, limits=None, points=None, polygons=None):
    """Discrete cells, nearest display enlargement, explicit color endpoints."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Invalid map for display')
    lo, hi = (float(a.min()), float(a.max())) if limits is None else limits
    scaled = np.clip((a-lo)/(hi-lo), 0., 1.) if hi > lo else np.zeros_like(a)
    rgb = np.stack((255*scaled, 200*np.sqrt(scaled), 255*(1-scaled)), axis=-1).astype(np.uint8)
    tile = Image.new('RGB', (320, 358), (20, 23, 28))
    # One uniform affine for cell pixels and overlays; native rectangular
    # patches retain aspect. Index0 lies at the center of the first cell.
    import cv2
    zoom = 288/max(a.shape)
    offset = np.array([(288-a.shape[1]*zoom)/2, (288-a.shape[0]*zoom)/2])
    affine = np.array([[zoom, 0, offset[0]+zoom/2-.5],
                       [0, zoom, offset[1]+zoom/2-.5]], dtype=float)
    enlarged = cv2.warpAffine(rgb, affine, (288, 288), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 23, 28))
    tile.paste(Image.fromarray(enlarged), (16, 42))
    draw = ImageDraw.Draw(tile); draw.text((12, 8), title, fill='white')
    draw.text((12, 335), 'color [%.4g, %.4g], %dx%d cells'%(lo, hi, a.shape[1], a.shape[0]), fill='white')
    colors = dict(gt=(90, 240, 110), b=(255, 180, 40),
                  center_only=(255, 100, 220), joint=(80, 210, 255))
    def display(xy):
        return (np.asarray(xy)+.5)*zoom+offset-.5+[16, 42]
    for name, polygon in (polygons or {}).items():
        p = display(polygon).tolist()
        draw.line([tuple(x) for x in p+[p[0]]], fill=colors[name], width=1)
    for name, xy in (points or {}).items():
        x, y = display(xy)
        if 16 <= x < 304 and 42 <= y < 330:
            draw.line((x-4, y, x+4, y), fill=colors[name], width=2)
            draw.line((x, y-4, x, y+4), fill=colors[name], width=2)
    return tile


def render_panel(row, native_pixels, path):
    from crane_project.utils import port_train_geometry_evidence_v1 as e
    polygons = {k:e.box_polygon(v) for k,v in row['boxes_model'].items()}
    bounds = e.crop_bounds(list(polygons.values()), list(native_pixels.size))
    centers = {k:np.asarray(v)[:2] for k,v in row['boxes_model'].items()}
    raw, mapping = e.crop_tile(native_pixels, bounds)
    overlay, other = e.crop_tile(native_pixels, bounds, polygons, centers)
    if mapping != other: raise ValueError('Pixel/overlay display affine differs')
    patch = row['p3_patch']; x0, y0 = patch['index_bounds_inclusive'][:2]
    native_centers = {k:(np.asarray(v)/SETTINGS['stride']-[x0, y0]).tolist() for k,v in centers.items()}
    native_polygons = {k:(v/SETTINGS['stride']-[x0, y0]).tolist() for k,v in polygons.items()}
    roi_centers = row['reference_roi_coordinates']['ordinary']
    ordinary = row['cached_roi']['ordinary']; aligned = row['cached_roi']['aligned']
    responses = row['responses']['shifts']
    diff_max = max(float(np.max(v['difference_map'])) for v in responses.values())
    tiles = [heat_tile(patch['maps']['channel_rms'], 'P3 native all-channel RMS', points=native_centers, polygons=native_polygons),
        heat_tile(patch['maps']['channel_spatial_deviation_rms'], 'P3 spatial deviation RMS', points=native_centers, polygons=native_polygons),
        heat_tile(ordinary['maps']['channel_rms'], 'Cached ordinary ROI RMS', points=roi_centers),
        heat_tile(aligned['maps']['channel_rms'], 'Cached aligned ROI RMS', points=row['reference_roi_coordinates']['aligned']),
        heat_tile(ordinary['maps']['channel_spatial_deviation_rms'], 'Ordinary spatial deviation RMS', points=roi_centers)]
    tiles += [heat_tile(responses[k]['difference_map'], k+' : shifted-minus-B RMS', (0., diff_max), roi_centers) for k in SHIFTS]
    panel = Image.new('RGB', (1600, 1160), (20, 23, 28)); draw = ImageDraw.Draw(panel)
    draw.text((12, 10), '%s / %s / %s / scale%s'%(row['image'], row['role'], row['domain'], row['scale']), fill='white')
    panel.paste(raw, (0, 40))
    panel.paste(overlay, (420, 40))
    draw.text((12, 30), 'Native input pixels: raw | overlay (display only)', fill='white')
    for i, tile in enumerate(tiles):
        # Five maps in row2, four fixed perturbation maps in row3.
        panel.paste(tile, ((i%5)*320, 400+(i//5)*358))
    draw.text((855, 65), 'B orange | GT green | center-only pink | joint cyan', fill='white')
    draw.text((855, 95), 'Energy maps are NOT object probabilities / correct-center scores.', fill='white')
    draw.text((855, 125), '9x9 bilinear ROI samples do NOT add native spatial resolution.', fill='white')
    draw.text((855, 155), 'Shift only ROI center +/-1 MODEL pixel; shape/context fixed.', fill='white')
    draw.text((855, 185), 'Four difference maps share one color range within this view.', fill='white')
    draw.text((855, 215), 'GT / saved centers are offline TRAIN references; no ranking.', fill='white')
    draw.text((855, 245), 'All channels measured; no selected-channel heatmap.', fill='white')
    draw.text((855, 275), 'Patch curves use stride8 coordinates, not receptive fields.', fill='white')
    panel.save(path)
    return dict(file=path.name, panel_wh=list(panel.size), pixel_crop_affine=mapping,
        display_note='Nearest-only heatmaps; per-map ranges printed. Cross-view colors are not comparable.',
        p3_index_bounds=patch['index_bounds_inclusive'])
