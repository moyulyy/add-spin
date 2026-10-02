"""Regression checks for POSCAR geometry and collinear magnetic initial guesses.

Run with ``python -m unittest discover -s tests -v``. Structures are generated
in memory so that production code does not need an example-data collection.
"""

import importlib.util
import itertools
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "add-spin.py"
SPEC = importlib.util.spec_from_file_location("add_spin_under_test", SCRIPT)
spin = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spin
SPEC.loader.exec_module(spin)


def poscar_text(lattice=None, coords=None, species=("Fe",), counts=None,
                scale="1.0", mode="Direct"):
    """Serialize only the small geometry needed by an individual test."""
    lattice = np.eye(3) * 2.87 if lattice is None else np.asarray(lattice)
    coords = [[0.0, 0.0, 0.0]] if coords is None else np.asarray(coords)
    counts = [len(coords)] if counts is None else counts
    rows = ["regression structure", scale]
    rows.extend(" ".join(map(str, row)) for row in lattice)
    rows.extend([" ".join(species), " ".join(map(str, counts)), mode])
    rows.extend(" ".join(map(str, row)) for row in coords)
    return "\n".join(rows) + "\n"


def bcc_conventional(order=(0, 1)):
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return spin.parse_poscar_text(poscar_text(coords=frac[list(order)]))


def nio_cell(repeat=1):
    """Build rocksalt conventional cells; magnetic periodicity is tested below."""
    nickel = np.array([[0, 0, 0], [0, 0.5, 0.5],
                       [0.5, 0, 0.5], [0.5, 0.5, 0]])
    oxygen = (nickel + [0.5, 0, 0]) % 1.0
    shifts = list(itertools.product(range(repeat), repeat=3))
    coords = np.concatenate([(sites + shift) / repeat
                             for sites in (nickel, oxygen) for shift in shifts])
    count = 4 * repeat ** 3
    return spin.parse_poscar_text(poscar_text(
        lattice=np.eye(3) * 4.17 * repeat, coords=coords,
        species=("Ni", "O"), counts=[count, count]))


def active_incar_assignments(text):
    """Read the simple key/value statements used by these output checks."""
    statements = []
    for line in text.splitlines():
        active = re.split(r"[#!]", line, maxsplit=1)[0]
        for statement in active.split(";"):
            if "=" in statement:
                key, value = statement.split("=", 1)
                statements.append((key.strip().upper(), value.strip()))
    return statements


class PoscarParsingTests(unittest.TestCase):
    def test_three_scale_factors_scale_cartesian_columns(self):
        lattice = np.array([[2.0, 0.5, 0.0], [0.0, 3.0, 0.5], [0.2, 0.0, 4.0]])
        scale = np.array([2.0, 3.0, 4.0])
        raw_cart = np.array([[0.4, 0.6, 0.8]])
        pos = spin.parse_poscar_text(poscar_text(
            lattice=lattice, coords=raw_cart, scale="2 3 4", mode="Cartesian"))
        np.testing.assert_allclose(pos.lattice, lattice * scale)
        np.testing.assert_allclose(pos.cartesian(), raw_cart * scale)
        np.testing.assert_allclose(pos.frac, raw_cart @ np.linalg.inv(lattice))

    def test_three_scale_factors_leave_direct_coordinates_unchanged(self):
        lattice = np.array([[2.0, 0.5, 0.0], [0.0, 3.0, 0.5], [0.2, 0.0, 4.0]])
        frac = np.array([[0.1, 0.2, 0.3]])
        pos = spin.parse_poscar_text(poscar_text(
            lattice=lattice, coords=frac, scale="2 3 4"))
        np.testing.assert_allclose(pos.lattice, lattice * [2.0, 3.0, 4.0])
        np.testing.assert_allclose(pos.frac, frac)

    def test_negative_single_scale_is_target_volume(self):
        pos = spin.parse_poscar_text(poscar_text(
            lattice=np.eye(3) * 2.0, coords=[[0.5, 0.0, 0.0]],
            scale="-64", mode="Cartesian"))
        self.assertAlmostEqual(pos.volume, 64.0)
        np.testing.assert_allclose(pos.cartesian(), [[1.0, 0.0, 0.0]])

    def test_rejects_invalid_scale_factors(self):
        for scale in ("", "0", "nan", "inf", "1 2", "1 2 3 4", "1 0 1", "1 -1 1", "1 nan 1"):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                spin.parse_poscar_text(poscar_text(scale=scale))

    def test_rejects_singular_or_nonfinite_lattice(self):
        lattices = [np.zeros((3, 3)), [[1, 0, 0], [2, 0, 0], [0, 0, 1]],
                    [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[1, 0, 0], [0, np.inf, 0], [0, 0, 1]]]
        for lattice in lattices:
            for scale in ("1", "-10"):
                with self.subTest(lattice=lattice, scale=scale), self.assertRaises(ValueError):
                    spin.parse_poscar_text(poscar_text(lattice=lattice, scale=scale))

    def test_rejects_nonfinite_coordinates(self):
        for mode in ("Direct", "Cartesian"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(mode=mode, value=value), self.assertRaises(ValueError):
                    spin.parse_poscar_text(poscar_text(coords=[[value, 0, 0]], mode=mode))

    def test_rejects_invalid_counts_mode_and_incomplete_coordinates(self):
        malformed = [
            poscar_text(counts=[0]),
            poscar_text(counts=[-1]),
            poscar_text(species=("Fe", "O"), counts=[1]),
            poscar_text(species=("Fe",), counts=[1, 1]),
            poscar_text(mode="Unrecognized"),
            poscar_text(coords=[[0, 0]]),
            poscar_text(counts=[2]),
        ]
        for text in malformed:
            with self.subTest(text=text), self.assertRaises(ValueError):
                spin.parse_poscar_text(text)


class SlabGeometryTests(unittest.TestCase):
    def test_vacuum_gaps_use_normal_heights_in_skew_cell(self):
        lattice = np.array([[2.0, 0, 0], [0, 2.0, 0], [12.0, 0, 4.0]])
        pos = spin.parse_poscar_text(poscar_text(lattice=lattice))
        gaps = spin.vacuum_gaps(pos)
        volume = abs(np.linalg.det(lattice))
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            height = volume / np.linalg.norm(np.cross(lattice[others[0]], lattice[others[1]]))
            self.assertAlmostEqual(gaps["abc"[axis]], height)
        self.assertAlmostEqual(gaps["c"], 4.0)
        self.assertEqual(spin.detect_system_type(pos)[0], "bulk")

    def test_slab_layers_cross_boundary_and_allow_every_vacuum_axis(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                lattice = np.eye(3) * 2.0
                lattice[axis, axis] = 20.0
                # Three layers at normal heights -1, 0, +1 angstrom, with
                # deliberately unsorted atom order and a periodic boundary.
                layer_ids = np.array([2, 0, 1, 2, 1, 0])
                frac = np.zeros((6, 3))
                frac[:, axis] = np.array([0.95, 0.0, 0.05])[layer_ids]
                frac[:, (axis + 1) % 3] = [0, 0, 0, 0.5, 0.5, 0.5]
                pos = spin.parse_poscar_text(poscar_text(lattice=lattice, coords=frac))
                self.assertEqual(spin.detect_system_type(pos)[0], "slab")
                result = spin.slab_layer_analysis(pos)
                self.assertEqual(result["vacuum_axis"], "abc"[axis])
                self.assertEqual(result["n_layers"], 3)
                self.assertAlmostEqual(result["slab_thickness"], 2.0)
                self.assertAlmostEqual(spin.vacuum_gaps(pos)["abc"[axis]], 18.0)
                np.testing.assert_allclose(np.diff(result["layer_positions"]), [1.0, 1.0])
                labels = np.asarray(result["layers"])
                np.testing.assert_array_equal(
                    labels[:, None] == labels[None, :],
                    layer_ids[:, None] == layer_ids[None, :])
                np.testing.assert_allclose(result["layer_depth"], (layer_ids == 1).astype(float))

                # Moving sites by lattice translations or reordering atoms
                # must preserve the geometric layer partition.
                permutation = np.array([4, 0, 5, 2, 1, 3])
                translated = frac[permutation] + np.array([1.0, -2.0, 3.0])
                moved = spin.parse_poscar_text(poscar_text(lattice=lattice, coords=translated))
                moved_result = spin.slab_layer_analysis(moved)
                moved_labels = np.asarray(moved_result["layers"])
                expected = layer_ids[permutation]
                np.testing.assert_array_equal(
                    moved_labels[:, None] == moved_labels[None, :],
                    expected[:, None] == expected[None, :])
                self.assertAlmostEqual(moved_result["slab_thickness"], 2.0)


class MagneticPlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.pos = bcc_conventional()

    def plan(self, **changes):
        args = {"assignments": [{"element": "Fe", "moment": 2.2}],
                "magnetic_order": "fm", "ispin": 2}
        args.update(changes)
        return args

    def test_valid_ferromagnetic_plan(self):
        moments, warnings = spin.plan_from_llm_args(self.pos, {}, self.plan())
        np.testing.assert_allclose(moments, [2.2, 2.2])
        self.assertEqual(warnings, [])

    def test_rejects_missing_assignment_fields_and_nonfinite_moments(self):
        invalid = [[{"moment": 2.2}], [{"element": "Fe"}],
                   [{"element": "Fe", "moment": float("nan")}],
                   [{"element": "Fe", "moment": float("inf")}]]
        for assignments in invalid:
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                spin.plan_from_llm_args(self.pos, {}, self.plan(assignments=assignments))

    def test_rejects_uncovered_chemical_element(self):
        pos = spin.parse_poscar_text(poscar_text(
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]], species=("Fe", "O"), counts=[1, 1]))
        with self.assertRaises(ValueError):
            spin.plan_from_llm_args(pos, {}, self.plan())

    def test_rejects_out_of_range_override(self):
        for index in (-1, self.pos.n_atoms):
            with self.subTest(index=index), self.assertRaises(ValueError):
                spin.plan_from_llm_args(self.pos, {}, self.plan(
                    site_overrides=[{"index": index, "moment": 1.0}]))

    def test_rejects_nonfinite_override(self):
        with self.assertRaises(ValueError):
            spin.plan_from_llm_args(self.pos, {}, self.plan(
                site_overrides=[{"index": 0, "moment": float("nan")}]))

    def test_ispin_one_rejects_nonzero_moments(self):
        with self.assertRaises(ValueError):
            spin.plan_from_llm_args(self.pos, {}, self.plan(ispin=1))
        with self.assertRaises(ValueError):
            spin.format_magmom(self.pos, [2.2, 2.2], ispin=1)

    def test_ispin_one_accepts_zero_moments(self):
        text = spin.format_magmom(self.pos, [0.0, 0.0], ispin=1)
        self.assertIn("ISPIN = 1", text)
        self.assertNotIn("MAGMOM =", text)

    def test_zero_moments_default_to_ispin_one(self):
        text = spin.format_magmom(self.pos, [0.0, 0.0])
        self.assertEqual(active_incar_assignments(text), [("ISPIN", "1")])

    def test_format_rejects_nonfinite_moments(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spin.format_magmom(self.pos, [value, 2.2])

    def test_invalid_tool_submission_does_not_set_submitted(self):
        runtime = spin.ToolRuntime(self.pos)
        invalid = self.plan(assignments=[{"element": "Fe"}])
        result = runtime.call("submit_magmom", invalid)
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("error"))
        self.assertIsNone(runtime.submitted)

        valid = self.plan()
        preview = runtime.call("preview_magmom", valid)
        self.assertTrue(preview["ok"])
        self.assertIsNone(runtime.submitted)
        self.assertTrue(runtime.call("submit_magmom", valid)["ok"])
        self.assertEqual(runtime.submitted, valid)
        self.assertFalse(runtime.call("submit_magmom", invalid)["ok"])
        self.assertEqual(runtime.submitted, valid)


class IncarUpdateTests(unittest.TestCase):
    def test_replaces_continued_magmom_without_leaving_orphan_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "INCAR"
            path.write_text("ENCUT = 520\nMAGMOM = 1*5 \\\n  1*-5\nISPIN = 2\n", encoding="utf-8")
            spin.append_to_incar(spin.format_magmom(bcc_conventional(), [2.2, 2.2]), str(path))
            result = path.read_text(encoding="utf-8")
            self.assertNotIn("1*-5", result)
            self.assertEqual(dict(active_incar_assignments(result))["MAGMOM"], "2*2.2")
            self.assertIn("ENCUT = 520", result)

    def test_unterminated_continuation_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "INCAR"
            old = b"MAGMOM = 2*5 \\\n"
            path.write_bytes(old)
            with self.assertRaises(ValueError):
                spin.append_to_incar(spin.format_magmom(bcc_conventional(), [2.2, 2.2]), str(path))
            self.assertEqual(path.read_bytes(), old)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="add-spin-test-")
        self.addCleanup(self.directory.cleanup)
        self.incar = Path(self.directory.name) / "INCAR"
        self.block = spin.format_magmom(bcc_conventional(), [2.2, -2.2])

    def test_replaces_existing_spin_tags_and_preserves_semicolon_parameters(self):
        self.incar.write_text(
            "ENCUT = 520; ISPIN = 1; PREC = Accurate\n"
            "ispin = 2; MAGMOM = 2*5; SIGMA = 0.05 ! smearing\n"
            "MAGMOM = 2*4\n"
            "# MAGMOM = 99 is an inactive note\n", encoding="utf-8")
        for _ in range(2):
            spin.append_to_incar(self.block, str(self.incar))
            text = self.incar.read_text(encoding="utf-8")
            pairs = active_incar_assignments(text)
            for tag in ("ISPIN", "MAGMOM"):
                self.assertEqual(sum(key == tag for key, _ in pairs), 1)
            values = dict(pairs)
            self.assertEqual(values["ISPIN"], "2")
            np.testing.assert_allclose(list(map(float, values["MAGMOM"].split())), [2.2, -2.2])
            self.assertEqual(values["ENCUT"], "520")
            self.assertEqual(values["PREC"], "Accurate")
            self.assertEqual(values["SIGMA"], "0.05")
            self.assertIn("! smearing", text)
            self.assertIn("# MAGMOM = 99 is an inactive note", text)

    def test_nonmagnetic_update_removes_previous_magmom(self):
        self.incar.write_text("ISPIN = 2; MAGMOM = 2*5; ENCUT = 520\n", encoding="utf-8")
        spin.append_to_incar(spin.format_magmom(bcc_conventional(), [0.0, 0.0]), str(self.incar))
        pairs = active_incar_assignments(self.incar.read_text(encoding="utf-8"))
        self.assertEqual(dict(pairs), {"ISPIN": "1", "ENCUT": "520"})

    def test_incompatible_existing_settings_reject_without_changing_file(self):
        settings = ["LSORBIT = .TRUE.", "LSORBIT = T", "lnoncollinear = true",
                    "LNONCOLLINEAR = .T.", "NUPDOWN = 0", "NUPDOWN = 2"]
        for setting in settings:
            with self.subTest(setting=setting):
                original = ("ENCUT = 520; ISPIN = 1\r\n"
                            f"MAGMOM = 2*5; {setting}; SIGMA = 0.05\r\n").encode("utf-8")
                self.incar.write_bytes(original)
                with self.assertRaises(ValueError):
                    spin.append_to_incar(self.block, str(self.incar))
                self.assertEqual(self.incar.read_bytes(), original)

    def test_inactive_or_disabled_incompatible_settings_allow_update(self):
        self.incar.write_text(
            "LSORBIT = .FALSE.; LNONCOLLINEAR = F; NUPDOWN = -1\n"
            "# LSORBIT = .TRUE.\n! NUPDOWN = 4\n", encoding="utf-8")
        spin.append_to_incar(self.block, str(self.incar))
        values = dict(active_incar_assignments(self.incar.read_text(encoding="utf-8")))
        self.assertEqual(values["LSORBIT"], ".FALSE.")
        self.assertEqual(values["LNONCOLLINEAR"], "F")
        self.assertEqual(values["NUPDOWN"], "-1")
        self.assertEqual(values["ISPIN"], "2")


class AntiferromagneticPeriodicityTests(unittest.TestCase):
    def test_default_selection_uses_all_sites(self):
        pos = bcc_conventional()
        self.assertEqual(spin.bipartite_signs(pos), spin.bipartite_signs(pos, indices=[0, 1]))

    def test_one_atom_primitive_cell_cannot_represent_afm(self):
        primitive = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) * 1.435
        pos = spin.parse_poscar_text(poscar_text(lattice=primitive))
        with self.assertRaises(ValueError):
            spin.bipartite_signs(pos, element="Fe")
        with self.assertRaises(ValueError):
            spin.plan_from_llm_args(pos, {}, {
                "assignments": [{"element": "Fe", "moment": 2.2}],
                "magnetic_order": "afm", "ispin": 2})

    def test_periodic_self_neighbor_prevents_false_afm_success(self):
        # Doubling just one bcc primitive vector still leaves nearest-neighbor
        # translations along the other two vectors that reverse the desired
        # spin, so this two-site simulation cell cannot host Neel bcc order.
        lattice = np.array([[-2, 2, 2], [1, -1, 1], [1, 1, -1]]) * 1.435
        pos = spin.parse_poscar_text(poscar_text(
            lattice=lattice, coords=[[0, 0, 0], [0.5, 0, 0]]))
        with self.assertRaises(ValueError):
            spin.bipartite_signs(pos, element="Fe")

    def test_compatible_bcc_afm_survives_atom_reordering(self):
        signs = spin.bipartite_signs(bcc_conventional(), element="Fe")
        original = np.array([signs[0], signs[1]])
        self.assertEqual(set(original), {-1, 1})
        reversed_signs = spin.bipartite_signs(bcc_conventional((1, 0)), element="Fe")
        reordered = np.array([reversed_signs[1], reversed_signs[0]])
        self.assertTrue(np.array_equal(original, reordered)
                        or np.array_equal(original, -reordered))

    def test_compatible_bcc_afm_plan_has_both_signs(self):
        moments, warnings = spin.plan_from_llm_args(bcc_conventional(), {}, {
            "assignments": [{"element": "Fe", "moment": 2.2}],
            "magnetic_order": "afm", "ispin": 2})
        np.testing.assert_allclose(sorted(moments), [-2.2, 2.2])
        self.assertEqual(warnings, [])

    def test_afm_graph_can_connect_different_elements(self):
        pos = spin.parse_poscar_text(poscar_text(
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]], species=("Fe", "Co"), counts=[1, 1]))
        for selection in ([], ["Fe", "Co"]):
            with self.subTest(selection=selection):
                moments, warnings = spin.plan_from_llm_args(pos, {}, {
                    "assignments": [{"element": "Fe", "moment": 2.0},
                                    {"element": "Co", "moment": 2.0}],
                    "magnetic_order": "afm", "afm_elements": selection})
                np.testing.assert_allclose(sorted(moments), [-2.0, 2.0])
                self.assertEqual(warnings, [])

    def test_zero_moment_site_is_excluded_from_afm_group(self):
        # The oxygen sits midway between the two magnetic sites. Including it
        # in the sign graph would split their bond and give a wrong spin pair.
        pos = spin.parse_poscar_text(poscar_text(
            coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
            species=("Fe", "O"), counts=[2, 1]))
        moments, warnings = spin.plan_from_llm_args(pos, {}, {
            "assignments": [{"element": "Fe", "moment": 2.2, "afm_group": "sites"},
                            {"element": "O", "moment": 0.0, "afm_group": "sites"}],
            "afm_elements": ["sites"], "magnetic_order": "afm"})
        np.testing.assert_allclose(sorted(moments[:2]), [-2.2, 2.2])
        self.assertEqual(moments[2], 0.0)
        self.assertEqual(warnings, [])

    def test_nio_conventional_cell_rejects_incompatible_afm_ii(self):
        pos = nio_cell()
        report = spin.analyze_structure(pos)
        self.assertEqual(report["motif"]["family"], "rocksalt")
        with self.assertRaises(ValueError):
            spin.plan_from_llm_args(pos, report, {
                "assignments": [{"element": "Ni", "moment": 2.0},
                                {"element": "O", "moment": 0.0}],
                "magnetic_order": "afm"})

    def test_nio_supercell_supports_compensated_afm_ii(self):
        pos = nio_cell(repeat=2)
        report = spin.analyze_structure(pos)
        self.assertEqual(report["motif"]["family"], "rocksalt")
        moments, warnings = spin.plan_from_llm_args(pos, report, {
            "assignments": [{"element": "Ni", "moment": 2.0},
                            {"element": "O", "moment": 0.0}],
            "magnetic_order": "afm"})
        nickel = np.array(moments[:32])
        np.testing.assert_allclose(np.abs(nickel), 2.0)
        self.assertEqual(np.count_nonzero(nickel > 0), 16)
        self.assertEqual(np.count_nonzero(nickel < 0), 16)
        np.testing.assert_allclose(moments[32:], 0.0)
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
