import os
import tempfile

import cv2
import numpy as np

from crane_project.tools.compare_dark_proxy_atlas_previews import (
    build_contact_sheet,
    discover_images,
    evenly_select,
)


def test_even_selection_covers_first_middle_and_last():
    paths = [f'image_{index:03d}.jpg' for index in range(11)]
    assert evenly_select(paths, 3) == [paths[0], paths[5], paths[10]]


def test_contact_sheet_discovers_and_renders_rows():
    with tempfile.TemporaryDirectory() as root:
        proxy_dir = os.path.join(root, 'proxy')
        atlas_dir = os.path.join(root, 'atlas')
        os.makedirs(proxy_dir)
        os.makedirs(atlas_dir)
        for index, directory in enumerate((proxy_dir, atlas_dir)):
            image = np.full(
                (80, 120, 3), 60 + 80 * index, dtype=np.uint8)
            assert cv2.imwrite(
                os.path.join(directory, f'frame_{index}.jpg'), image)

        proxy_paths = discover_images(proxy_dir)
        atlas_paths = discover_images(atlas_dir)
        assert len(proxy_paths) == 1
        assert len(atlas_paths) == 1
        sheet = build_contact_sheet([
            ('SOURCE PROXY', proxy_paths),
            ('TARGET-DEV ATLAS', atlas_paths),
        ], cell_width=180, cell_height=120)
        assert sheet.shape == (54 + 2 * 120, 300 + 180, 3)
        assert int(sheet.sum()) > 0
