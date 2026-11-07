import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import publish


def sample_pack(run: datetime, model: str = "ecmwf") -> dict:
    spec = publish.grid_spec(model)
    shape = (len(publish.hours(model, run)), spec["ny"], spec["nx"])
    lats = [round(publish.SOUTH + spec["spacing"] * j, 3) for j in range(spec["ny"])]
    lons = [round(publish.WEST + spec["spacing"] * i, 3) for i in range(spec["nx"])]
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

    def test_model_grids_are_independent(self):
        ecmwf = publish.grid_spec("ecmwf")
        gfs = publish.grid_spec("gfs")
        self.assertEqual(ecmwf, {"spacing": 0.1, "nx": 431, "ny": 341})
        self.assertEqual(gfs, {"spacing": 0.25, "nx": 173, "ny": 137})

    def test_ecmwf_indexes_include_surface_and_pressure_levels(self):
        run = datetime(2026, 11, 7, 0, tzinfo=timezone.utc)
        urls = publish.step_markers("ecmwf", run, 0)
        self.assertEqual(len(urls), 2)
        self.assertTrue(any("-sfc.index" in url and "/0p1/" in url for url in urls))
        self.assertTrue(any("-pl.index" in url and "/0p1/" in url for url in urls))
        gfs_urls = publish.step_markers("gfs", run, 0)
        self.assertEqual(len(gfs_urls), 1)
        self.assertIn("0p25", gfs_urls[0])

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

    def test_ecmwf_all_four_cycles_use_provider_native_horizons(self):
        for hour in (0, 12):
            run = datetime(2026, 11, 7, hour, tzinfo=timezone.utc)
            steps = publish.hours("ecmwf", run)
            self.assertEqual(publish.horizon_hours("ecmwf", run), 240)
            self.assertEqual(steps[-1], 240)
            self.assertEqual(len(steps), 65)
        for hour in (6, 18):
            run = datetime(2026, 11, 7, hour, tzinfo=timezone.utc)
            steps = publish.hours("ecmwf", run)
            self.assertEqual(publish.horizon_hours("ecmwf", run), 144)
            self.assertEqual(steps[-1], 144)
            self.assertEqual(len(steps), 49)

    def test_ecmwf_discovery_candidates_include_four_daily_cycles(self):
        now = datetime(2026, 11, 8, 9, tzinfo=timezone.utc)
        found = publish.candidates("ecmwf", now, lookback=4)
        self.assertEqual([run.hour for run in found], [6, 0, 18, 12])
        self.assertEqual([run.day for run in found], [8, 8, 7, 7])

    def test_same_run_with_legacy_144h_payload_is_not_treated_as_published(self):
        run = datetime(2026, 10, 30, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            publish.write_cycle(site, "ecmwf", sample_pack(run))
            publish.flip_pointer(site, "ecmwf", run, "test")
            self.assertTrue(publish.current_cycle_ok(site, "ecmwf", run))

            meta_path = publish.cycle_dir(site, "ecmwf", run) / "meta.json"
            meta = publish.read_json(meta_path)
            meta["horizonHours"] = 144
            meta["nT"] = 49
            meta_path.write_text(__import__("json").dumps(meta))

            current_path = site / "ecmwf" / "current.json"
            current = publish.read_json(current_path)
            current["horizonHours"] = 144
            current["nT"] = 49
            current_path.write_text(__import__("json").dumps(current))
            self.assertTrue(publish.pointer_matches(site, "ecmwf", run))
            self.assertFalse(publish.current_cycle_ok(site, "ecmwf", run))

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
                "nx": publish.grid_spec(model)["nx"],
                "ny": publish.grid_spec(model)["ny"],
                "gridSpacing": publish.grid_spec(model)["spacing"],
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

    def test_meta_accepts_native_144h_ecmwf_cycles(self):
        for hour in (6, 18):
            run = datetime(2026, 10, 30, hour, tzinfo=timezone.utc)
            pack = sample_pack(run, "ecmwf")
            files, digests = publish.bundle_files(pack)
            meta = publish.meta_of("ecmwf", pack, digests)
            self.assertEqual(meta["horizonHours"], 144)
            self.assertEqual(meta["nT"], 49)
            publish.validate_meta(meta, "ecmwf")

    def test_ecmwf_missing_gusts_are_nan_not_estimated_from_wind(self):
        fields = {
            "t2m": np.full((2, 3), 28.0, dtype=np.float32),
            "u10": np.full((2, 3), 12.0, dtype=np.float32),
            "v10": np.full((2, 3), 5.0, dtype=np.float32),
        }
        self.assertTrue(publish.mark_missing_ecmwf_gusts(fields))
        self.assertEqual(fields["gusts"].dtype, np.float32)
        self.assertTrue(np.isnan(fields["gusts"]).all())

    def test_ecmwf_native_gusts_are_preserved_unchanged(self):
        original = np.full((2, 3), 42.0, dtype=np.float32)
        fields = {"t2m": np.ones((2, 3), dtype=np.float32), "gusts": original.copy()}
        self.assertFalse(publish.mark_missing_ecmwf_gusts(fields))
        self.assertTrue(np.array_equal(fields["gusts"], original))

    def test_ecmwf_gust_mapping_accepts_only_native_10fg(self):
        self.assertEqual(publish.ecmwf_want({"param": "10fg", "levtype": "sfc"}), "gusts")
        self.assertIsNone(publish.ecmwf_want({"param": "gust", "levtype": "sfc"}))

    def test_gfs_analysis_precip_is_zeros_only(self):
        spec = publish.grid_spec("gfs")
        fields = {"t2m": np.ones((spec["ny"], spec["nx"]), dtype=np.float32)}
        geo = {"window": None}
        self.assertFalse(publish.fill_missing_analysis_precip("gfs", 3, fields, geo))
        self.assertNotIn("precip", fields)
        self.assertTrue(publish.fill_missing_analysis_precip("gfs", 0, fields, geo))
        self.assertEqual(float(fields["precip"].sum()), 0.0)
        self.assertEqual(geo["window"], (0, 0))
        self.assertFalse(publish.fill_missing_analysis_precip("ecmwf", 0, {"t2m": fields["t2m"]}, {"window": None}))


if __name__ == "__main__":
    unittest.main()
