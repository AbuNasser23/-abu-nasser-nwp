import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import publish


def sample_pack(run: datetime) -> dict:
    shape = (len(publish.hours()), publish.NY, publish.NX)
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
    return {"run": run, "lats": lats, "lons": lons, "fields": fields}


class PublishTests(unittest.TestCase):
    def test_horizon_stops_at_144(self):
        self.assertEqual(publish.hours()[0], 0)
        self.assertEqual(publish.hours()[-1], 144)
        self.assertNotIn(150, publish.hours())
        self.assertEqual(len(publish.hours()), 49)

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
            publish.write_cycle(site, "gfs", sample_pack(first))
            publish.flip_pointer(site, "gfs", first, "test")
            publish.write_cycle(site, "gfs", sample_pack(second))
            publish.flip_pointer(site, "gfs", second, "test")
            publish.prune(site, "gfs", 2)
            current = publish.read_json(site / "gfs" / "current.json")
            self.assertEqual(current["run"], "2026-10-30T00:00:00Z")
            self.assertEqual(current["previous"], "2026-10-29T12:00:00Z")
            names = sorted(path.name for path in (site / "gfs" / "cycles").iterdir())
            self.assertEqual(names, ["20261029T12Z", "20261030T00Z"])

    def test_rejected_meta_does_not_claim_240h(self):
        meta = {
            "model": "ecmwf",
            "nx": 173,
            "ny": 137,
            "nT": 65,
            "run": "2026-10-30T00:00:00Z",
            "times": [],
            "horizonHours": 240,
            "fields": list(publish.CORE),
        }
        with self.assertRaises(RuntimeError):
            publish.validate_meta(meta, "ecmwf")

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
