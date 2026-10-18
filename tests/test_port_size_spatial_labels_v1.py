import math
import pytest
from crane_project.tools import audit_port_size_spatial_labels_v1 as a


@pytest.mark.parametrize('center,angle', [((0,0),0),((10,-8),.2),((-40,10),-.9)])
def test_full_mass_intrinsic_size_invariant_to_pose(center, angle):
    p=a.rectangle(*center,100,50,angle)
    assert a.intrinsic_from_polygon(p)==pytest.approx([100,50],rel=1e-12)
    assert a.intrinsic_from_polygon(list(reversed(p)))==pytest.approx([100,50],rel=1e-12)


def test_translation_cancels_from_boundary_difference_not_label_alignment():
    x=a.inspect([10,5,100,50,0],[0,0,100,50,0])
    assert x['projected_widths']==pytest.approx([100,50])
    assert x['virtual_real_box_iou']<1
    assert x['true_fully_in_ideal_M_search']


def test_size_exact_angle_error_is_not_size_supervision():
    x=a.inspect([0,0,100,50,math.radians(5)],[0,0,100,50,0])
    assert x['projection_relative_errors'][1]>.17
    assert max(map(abs,x['full_moment_relative_errors']))<1e-12
    assert x['virtual_real_box_iou']<1


def test_clipping_or_absent_mass_cannot_be_full_rectangle_moments():
    x=a.inspect([0,0,100,50,math.pi/4],[0,0,100,50,0])
    assert not x['true_fully_in_ideal_M_search'] and x['virtual_fully_in_ideal_M_search']
    assert x['true_area_fraction_in_ideal_M_search']<1
    assert max(map(abs,x['clipped_moment_relative_errors']))>.01
    empty=a.inspect([500,500,100,50,0],[0,0,100,50,0])
    assert empty['true_area_fraction_in_ideal_M_search']==0
    assert empty['clipped_moment_relative_errors'] is None


def test_binary_alias_is_not_a_claim_about_fractional_labels():
    assert a.point_mask([0,0,94,40,0])==a.point_mask([0,0,96,41,0])
    p=a.rectangle(0,0,94,40,0);q=a.rectangle(0,0,96,41,0)
    assert a.area(p)!=a.area(q)


def test_clipping_area_and_covariance_pixel_coordinates():
    p=a.rectangle(0,0,100,50,0)
    assert a.area(a.clip_rect(p,0,50,-25,25))==pytest.approx(2500)
    assert a.intrinsic_from_polygon(a.clip_rect(p,0,50,-25,25))==pytest.approx([50,50])
    # Eigenvalues in anisotropically normalized coordinates cannot be read as
    # physical edge lengths. Map points/covariance back before eigendecomposition.
    stretched=[(2*x,.5*y) for x,y in p]
    assert a.intrinsic_from_polygon(stretched)==pytest.approx([200,25])
    restored=[(x/2,y/.5) for x,y in stretched]
    assert a.intrinsic_from_polygon(restored)==pytest.approx([100,50])


def test_missing_views_and_conditional_moment_denominators():
    row=dict(geometry=a.inspect([500,500,100,50,0],[0,0,100,50,0]))
    s=a.summarize([row,dict(geometry=None)])
    assert s['views']==2 and s['outputs']==1 and s['missing']==1
    assert s['empty_true_target']==1 and s['long_axis_clipped_moment_evaluable']==0
    assert s['long_axis_clipped_moment_MAE_pct'] is None
