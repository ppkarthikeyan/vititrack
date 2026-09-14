import numpy as np
from vititrack.eval.quality import mask_flags

def test_flags():
    m = np.zeros((100, 100), bool); m[10:50, 10:50] = True
    assert mask_flags(m)["boxy"]
    m2 = np.zeros((100, 100), bool); yy, xx = np.mgrid[:100, :100]; m2[(xx-50)**2 + (yy-50)**2 < 20**2] = True
    f = mask_flags(m2); assert not f["boxy"] and not f["suspect"]
    assert mask_flags(np.ones((100, 100), bool))["whole"]
    assert mask_flags(np.zeros((100, 100), bool))["empty"]
