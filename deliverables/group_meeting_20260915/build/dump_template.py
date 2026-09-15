# -*- coding: utf-8 -*-
"""只读：把模板每页的形状（名称/类型/几何/文字/字号）导出成文本，供人工核对。
不修改任何文件。用法：python3 dump_template.py > template_dump.txt
"""
import sys
from pptx import Presentation
from pptx.util import Emu

EMU_PER_IN = 914400.0


def inch(v):
    if v is None:
        return None
    return round(v / EMU_PER_IN, 3)


def walk(shapes, depth, out):
    for sh in shapes:
        try:
            if sh.shape_type == 6 and sh.__class__.__name__ == 'GroupShape':
                out.append('%s[GROUP] %s' % ('  ' * depth, sh.name))
                walk(sh.shapes, depth + 1, out)
                continue
        except Exception:
            pass
        txt = ''
        sizes = set()
        if sh.has_text_frame:
            txt = sh.text_frame.text.replace('\n', '\\n')
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if r.font.size is not None:
                        sizes.add(r.font.size.pt)
        out.append('%s%s | type=%s | L=%s T=%s W=%s H=%s | sizes=%s | len=%d | %s' % (
            '  ' * depth, sh.name, sh.shape_type,
            inch(sh.left), inch(sh.top), inch(sh.width), inch(sh.height),
            sorted(sizes), len(txt), txt[:180]))


def main():
    prs = Presentation(sys.argv[1])
    out = []
    out.append('slide_count=%d size=%sx%s in' % (
        len(prs.slides), inch(prs.slide_width), inch(prs.slide_height)))
    for i, slide in enumerate(prs.slides, 1):
        out.append('')
        out.append('=' * 100)
        out.append('SLIDE %d  layout=%s' % (i, slide.slide_layout.name))
        out.append('=' * 100)
        walk(slide.shapes, 0, out)
        if slide.has_notes_slide:
            nt = slide.notes_slide.notes_text_frame.text
            out.append('  [NOTES] len=%d %s' % (len(nt), nt[:200]))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
