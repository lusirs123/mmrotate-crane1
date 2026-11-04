"""Prepare an initialization source, without changing the official OBB model."""
from copy import deepcopy


def backbone_source(classifier, detector):
    """Retain the nine architecture-identical ImageNet backbone blocks.

    The classifier's block 9 is C2PSA while detection block 9 is SPPF. Some
    BatchNorm tensors share names/shapes despite different semantics, so loading
    the whole classifier checkpoint would also copy unrelated tensors there.
    The official native .load() remains responsible for weight transfer.
    """
    if classifier.model[-1].__class__.__name__ != 'Classify' or classifier.yaml['nc'] != 1000:
        raise ValueError('Expected official 1000-class ImageNet classifier')
    if classifier.yaml['backbone'][:9] != detector.yaml['backbone'][:9]:
        raise ValueError('Classification and OBB backbone definitions differ')
    if classifier.yaml['scale'] != 'm' or detector.yaml['scale'] != 'm':
        raise ValueError('Expected medium-scale source and target')
    for i in range(9):
        source, target = classifier.model[i], detector.model[i]
        if source.__class__ != target.__class__:
            raise ValueError('Shared block class differs')
        a, b = source.state_dict(), target.state_dict()
        if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
            raise ValueError('Shared backbone tensor layout differs')
    result = deepcopy(classifier)
    result.model = result.model[:9]
    return result
