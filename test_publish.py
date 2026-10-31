import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import publish


def sample_pack(run: datetime, model: str = "ecmwf") -> dict:
    shape = (len(publish.hours(model)), publish.NY, publish.NX)
    lats = [round(publish.SOUTH + 0.25 * j, 2) for j in range(publish.NY)]
    lons = [round(publish.WEST + 0.25 * i, 2) for i in range(publish.NX)]
    fields = {name: np.zeros(shape, dtype=np.float32) for name in publish.CORE}
    fields["t2m"][:] = 28
    fields["td2m"][:] = 14
    fields["rh2m"][:] = 40
    fields["mslp"][:] = 1010
    fields["z500"][:] = 5880
    fields["z850"][:] = 1500
    fields["z700"][:] = 3100
    fields["z300"][:] = 9600
    fields["u10"][:] = 12
    fields["lit"] = np.full(shape, np.nan, dtype=np.float32)
    fields["lifted"] = np.full(shape, np.nan, dtype=np.float32)
    return {"model": model, "run": run, "lats": lats, "lons": lons, "fields": fields}


class PublishTests(unittest.TestCase):
    def test_ecmwf_native_horizon_is_240h(self):
        steps = publish.hours("ecmwf")
        self.assertEqual(steps[0], 0)
        self.assertEqual(steps[-1], 240)
        self.assertEqual(len(steps), 65)
        self.assertEqual(steps[:49], list(range(0, 145, 3)))
        self.assertEqual(steps[49:], list(range(150, 241, 6)))

    def test_gfs_native_horizon_is_240h(self):
        steps = publish.hours("gfs")
        self.assertEqual(steps[0], 0)
        self.assertEqual(steps[-1], 240)
        self.assertEqual(len(steps), 61)
        self.assertEqual(steps[:41], list(range(0, 121, 3)))
        self.assertEqual(steps[41:], list(range(126, 241, 6)))
        self.assertNotIn(123, steps)
        self.assertIn(144, steps)

    def test_pointer_stays_until_cycle_is_readable(self):
        run = datetime(2026, 10, 30, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            publish.write_cycle(site, "ecmwf", sample_pack(run))
            self.assertIsNone(publish.read_json(site / "ecmwf" / "current.json"))
            broken = publish.cycle_dir(site, "ecmwf", run) / "t2m.f32.bin"
            broken.unlink()
            with self.assertRaises(RuntimeError):
                publish.flip_pointer(site, "ecmwf", run, "test")
            self.assertFalse((site / "ecmwf" / "current.json").exists())

    def test_second_cycle_keeps_the_previous_pointer(self):
        first = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
        second = datetime(2026, 10, 30, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            publish.write_cycle(site, "gfs", sample_pack(first, "gfs"))
            publish.flip_pointer(site, "gfs", first, "test")
            publish.write_cycle(site, "gfs", sample_pack(second, "gfs"))
            publish.flip_pointer(site, "gfs", second, "test")
            publish.prune(site, "gfs", 2)
            current = publish.read_json(site / "gfs" / "current.json")
            self.assertEqual(current["run"], "2026-10-30T00:00:00Z")
            self.assertEqual(current["previous"], "2026-10-29T12:00:00Z")
            names = sorted(path.name for path in (site / "gfs" / "cycles").iterdir())
            self.assertEqual(names, ["20261029T12Z", "20261030T00Z"])

    def test_meta_accepts_240h_and_rejects_144h(self):
        run = datetime(2026, 10, 30, 0, tzinfo=timezone.utc)
        for model in ("ecmwf", "gfs"):
            meta = {
                "model": model,
                "nx": publish.NX,
                "ny": publish.NY,
                "nT": len(publish.hours(model)),
                "run": publish.iso_of(run),
                "times": publish.expected_times(run, model),
                "horizonHours": 240,
                "fields": list(publish.CORE),
            }
            publish.validate_meta(meta, model)
            meta["horizonHours"] = 144
            with self.assertRaises(RuntimeError):
                publish.validate_meta(meta, model)

    def test_ecmwf_missing_gusts_gets_explicit_wind_based_estimate(self):
        fields = {
            "u10": np.full((2, 3), 12.0, dtype=np.float32),
            "v10": np.full((2, 3), 5.0, dtype=np.float32),
        }
        self.assertTrue(publish.fill_missing_ecmwf_gusts(fields))
        expected = round((13.0 * 1.3) * 10) / 10
        self.assertTrue(np.allclose(fields["gusts"], expected))
        self.assertEqual(fields["gusts"].dtype, np.float32)

    def test_ecmwf_provider_gusts_are_never_overwritten(self):
        original = np.full((2, 3), 42.0, dtype=np.float32)
        fields = {
            "u10": np.full((2, 3), 12.0, dtype=np.float32),
            "v10": np.full((2, 3), 5.0, dtype=np.float32),
            "gusts": original.copy(),
        }
        self.assertFalse(publish.fill_missing_ecmwf_gusts(fields))
        self.assertTrue(np.array_equal(fields["gusts"], original))

    def test_missing_gusts_without_wind_components_is_not_fabricated(self):
        fields = {"u10": np.ones((2, 3), dtype=np.float32)}
        self.assertFalse(publish.fill_missing_ecmwf_gusts(fields))
        self.assertNotIn("gusts", fields)

    def test_gfs_analysis_precip_is_zeros_only(self):
        fields = {"t2m": np.ones((publish.NY, publish.NX), dtype=np.float32)}
        geo = {"window": None}
        self.assertFalse(publish.fill_missing_analysis_precip("gfs", 3, fields, geo))
        self.assertNotIn("precip", fields)
        self.assertTrue(publish.fill_missing_analysis_precip("gfs", 0, fields, geo))
        self.assertEqual(float(fields["precip"].sum()), 0.0)
        self.assertEqual(geo["window"], (0, 0))
        self.assertFalse(publish.fill_missing_analysis_precip("ecmwf", 0, {"t2m": fields["t2m"]}, {"window": None}))


if __name__ == "__main__":
    unittest.main()
