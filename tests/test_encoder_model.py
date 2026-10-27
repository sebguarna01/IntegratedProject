import numpy as np

from activenav.sensing.odometry import EncoderErrorModel

CLEAN = dict(encoder_ticks_per_rev=0, encoder_mult_std=0.0, encoder_add_std=0.0,
             wheel_scale_std=0.0, track_width_error_std=0.0)


def _ramp(n=200):
    return np.linspace(0.0, 10.0, n)[:, None] * np.array([[1.0, 1.0]])


def test_clean_model_is_identity():
    m = EncoderErrorModel(CLEAN, np.random.default_rng(0))
    m.reset(np.zeros(2))
    out = np.array([m.measure(a) for a in _ramp()])
    assert np.allclose(out, _ramp())


def test_quantisation_is_multiple_of_step():
    cfg = dict(CLEAN, encoder_ticks_per_rev=4096)
    m = EncoderErrorModel(cfg, np.random.default_rng(0))
    m.reset(np.zeros(2))
    q = 2 * np.pi / 4096
    out = np.array([m.measure(a) for a in _ramp()])
    assert np.allclose(out / q, np.round(out / q))
    assert np.max(np.abs(out - _ramp())) <= q / 2 + 1e-12


def test_wheel_scale_error_scales_distance_and_is_constant_per_trial():
    cfg = dict(CLEAN, wheel_scale_std=0.01)
    m = EncoderErrorModel(cfg, np.random.default_rng(3))
    m.reset(np.zeros(2))
    s = m.scale.copy()
    out = np.array([m.measure(a) for a in _ramp()])
    assert np.allclose(out[-1], 10.0 * s)
    assert not np.allclose(s, 1.0)


def test_same_seed_same_errors_and_different_seed_different():
    cfg = dict(CLEAN, wheel_scale_std=0.01, track_width_error_std=0.01, encoder_mult_std=0.01)

    def run(seed):
        m = EncoderErrorModel(cfg, np.random.default_rng(seed))
        m.reset(np.zeros(2))
        return np.array([m.measure(a) for a in _ramp()])

    assert np.allclose(run(1), run(1))
    assert not np.allclose(run(1), run(2))


def test_track_error_changes_yaw_not_forward_distance():
    cfg = dict(CLEAN, track_width_error_std=0.05)
    m = EncoderErrorModel(cfg, np.random.default_rng(5))
    m.reset(np.zeros(2))
    true = np.array([1.0, 3.0])              # forward 2.0, differential 1.0
    out = m.measure(true)
    assert np.isclose(out.mean(), true.mean())
    assert not np.isclose(out[1] - out[0], true[1] - true[0])
