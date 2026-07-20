from scripts.tau2_estimator import fit_time_model, invert_duration

def test_fit_and_invert_recovers_osl():
    # synthetic ground truth: dur = 0.2 + 0.0001*isl + 0.01*osl
    rows = [{"isl": isl, "osl": osl, "dur": 0.2 + 0.0001*isl + 0.01*osl}
            for isl in (500, 1000, 2000) for osl in (100, 500, 2000, 5000)]
    m = fit_time_model(rows)
    assert abs(m["itl"] - 0.01) < 1e-3
    got = invert_duration(0.2 + 0.0001*1000 + 0.01*654, isl=1000, model=m)
    assert abs(got - 654) <= 5
