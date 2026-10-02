"""Physical seed conventions, offline configuration, and CLI failure behavior."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_add_spin import nio_cell, poscar_text, spin


class IonicAndMolecularPhysicsTests(unittest.TestCase):
    def test_cuo_without_knowledge_base_uses_cu_two_plus_spin(self):
        pos = spin.parse_poscar_text(poscar_text(
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            species=("Cu", "O"), counts=[1, 1]))
        with patch.dict(spin.KNOWLEDGE_BASE, {}, clear=True):
            moments, reason = spin.heuristic_plan(pos)
        np.testing.assert_allclose(moments, [1.0, 0.0])
        self.assertNotIn("命中知识库", reason)

    def test_cuprite_copper_one_plus_is_closed_shell(self):
        coords = [[0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
                  [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
                  [0, 0, 0], [0.5, 0.5, 0.5]]
        pos = spin.parse_poscar_text(poscar_text(
            lattice=np.eye(3) * 4.27, coords=coords,
            species=("Cu", "O"), counts=[4, 2]))
        moments, _ = spin.heuristic_plan(pos)
        np.testing.assert_allclose(moments, [0.0] * 6)

    def test_closed_shell_perovskite_ions_have_zero_seeds(self):
        coords = [[0, 0, 0], [0.5, 0.5, 0.5],
                  [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        for species in (("Sr", "Ti", "O"), ("La", "Al", "O")):
            with self.subTest(species=species):
                pos = spin.parse_poscar_text(poscar_text(
                    lattice=np.eye(3) * 3.9, coords=coords,
                    species=species, counts=[1, 1, 3]))
                moments, _ = spin.heuristic_plan(pos)
                np.testing.assert_allclose(moments, [0.0] * 5)

    def test_oxygen_molecule_has_triplet_seed(self):
        pos = spin.parse_poscar_text(poscar_text(
            lattice=np.eye(3) * 20.0,
            coords=[[0.5, 0.5, 0.47], [0.5, 0.5, 0.53]], species=("O",)))
        report = spin.analyze_structure(pos)
        self.assertEqual(report["system_type"], "mole")
        moments, _ = spin.heuristic_plan(pos, report)
        np.testing.assert_allclose(moments, [1.0, 1.0])
        self.assertAlmostEqual(sum(moments), 2.0)

    def test_isolated_platinum_does_not_inherit_bulk_nonmagnetic_seed(self):
        pos = spin.parse_poscar_text(poscar_text(
            lattice=np.eye(3) * 20.0, coords=[[0.5, 0.5, 0.5]], species=("Pt",)))
        report = spin.analyze_structure(pos)
        self.assertEqual(report["system_type"], "mole")
        with self.assertRaises(ValueError):
            spin.heuristic_plan(pos, report)

    def test_octahedral_d_two_d_three_d_eight_low_spin_counts(self):
        for symbol, oxidation, d_count, unpaired in (
                ("Ti", 2, 2, 2), ("Cr", 3, 3, 3), ("Ni", 2, 8, 2)):
            with self.subTest(symbol=symbol, oxidation=oxidation):
                info = spin.element_info(symbol)
                ion = info["magnetic_ions"][f"{symbol}{oxidation}+"]
                self.assertEqual(info["electron_shell"], "d")
                self.assertEqual(ion["d_electrons"], d_count)
                self.assertEqual(ion["unpaired_electrons_low_spin"], unpaired)
                self.assertEqual(ion["unpaired_electrons_high_spin"], unpaired)

    def test_f_shell_counts_are_labeled_as_f_and_explain_soc_limit(self):
        for symbol, oxidation, count in (("Gd", 3, 7), ("Eu", 3, 6)):
            with self.subTest(symbol=symbol):
                info = spin.element_info(symbol)
                ion = info["magnetic_ions"][f"{symbol}{oxidation}+"]
                self.assertEqual(info["electron_shell"], "f")
                self.assertEqual(ion["f_electrons"], count)
                self.assertNotIn("d_electrons", ion)
                self.assertIn("SOC", info["spin_count_convention"])

    def test_fixed_oxidation_states_cannot_silently_violate_charge_balance(self):
        for composition in ({"Na": 1, "O": 2}, {"Na": 2, "O": 2}, {"H": 2, "O": 2}):
            with self.subTest(composition=composition), self.assertRaises(ValueError):
                spin._ionic_seed_moments(composition)
        moments, resolved = spin._ionic_seed_moments({"Na": 2, "O": 1})
        self.assertTrue(resolved)
        self.assertEqual(moments, {"Na": 0.0, "O": 0.0})


class LlmConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.header = patch.multiple(
            spin, LLM_API_KEY="header-test-key",
            LLM_BASE_URL="https://header.invalid/v1", LLM_MODEL="header-model",
            LLM_TEMPERATURE=0.3, LLM_TIMEOUT=91, LLM_MAX_RETRIES=4, LLM_MAX_STEPS=9)
        self.header.start()
        self.addCleanup(self.header.stop)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        no_network = patch.object(spin.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden in unit tests"))
        self.urlopen = no_network.start()
        self.addCleanup(no_network.stop)

    def test_header_configuration_is_used_by_default(self):
        args = spin.parse_args([])
        client = spin.build_client_from_args(args)
        self.assertEqual(client.api_key, "header-test-key")
        self.assertEqual(client.base_url, "https://header.invalid/v1")
        self.assertEqual(client.model, "header-model")
        self.assertEqual(client.temperature, 0.3)
        self.assertEqual(client.timeout, 91)
        self.assertEqual(client.max_retries, 4)
        self.assertEqual(args.max_steps, 9)
        self.urlopen.assert_not_called()

    def test_environment_overrides_header_connection_settings(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "env-test-key",
                                    "LLM_BASE_URL": "https://env.invalid/v1/",
                                    "LLM_MODEL": "env-model"}):
            client = spin.build_client_from_args(spin.parse_args([]))
        self.assertEqual(client.api_key, "env-test-key")
        self.assertEqual(client.base_url, "https://env.invalid/v1")
        self.assertEqual(client.model, "env-model")
        self.urlopen.assert_not_called()

    def test_cli_overrides_environment_and_header_configuration(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "env-test-key",
                                    "LLM_BASE_URL": "https://env.invalid/v1",
                                    "LLM_MODEL": "env-model"}):
            args = spin.parse_args([
                "--api-key", "cli-test-key", "--base-url", "https://cli.invalid/v1/",
                "--model", "cli-model", "--temperature", "0.7", "--timeout", "12",
                "--max-retries", "2", "--max-steps", "3"])
            client = spin.build_client_from_args(args)
        self.assertEqual(client.api_key, "cli-test-key")
        self.assertEqual(client.base_url, "https://cli.invalid/v1")
        self.assertEqual(client.model, "cli-model")
        self.assertEqual(client.temperature, 0.7)
        self.assertEqual(client.timeout, 12)
        self.assertEqual(client.max_retries, 2)
        self.assertEqual(args.max_steps, 3)
        self.urlopen.assert_not_called()

    def test_openai_and_deepseek_environment_key_fallbacks(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-key",
                                    "DEEPSEEK_API_KEY": "deepseek-test-key",
                                    "OPENAI_BASE_URL": "https://openai.invalid/v1"}):
            client = spin.build_client_from_args(spin.parse_args([]))
            self.assertEqual(client.api_key, "openai-test-key")
            self.assertEqual(client.base_url, "https://openai.invalid/v1")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-test-key"}):
            client = spin.build_client_from_args(spin.parse_args([]))
            self.assertEqual(client.api_key, "deepseek-test-key")
        self.urlopen.assert_not_called()


class CliFailureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="add-spin-cli-test-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.poscar = self.directory / "POSCAR"
        self.incar = self.directory / "INCAR"
        no_network = patch.object(spin.urllib.request, "urlopen", side_effect=AssertionError("Network forbidden in unit tests"))
        self.urlopen = no_network.start()
        self.addCleanup(no_network.stop)

    def test_help_has_no_example_export_or_embedded_self_test(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as result:
            spin.parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertNotIn("--make-examples", output.getvalue())
        self.assertNotIn("--self-test", output.getvalue())

    def assert_rejected_without_write(self, text):
        self.poscar.write_text(text, encoding="utf-8")
        for existing in (False, True):
            with self.subTest(existing_incar=existing):
                original = b"ENCUT = 520\r\nISPIN = 1\r\n"
                if existing:
                    self.incar.write_bytes(original)
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr), patch.object(
                        spin, "_ACTIVE_VACUUM_THRESHOLD", spin.DEFAULT_VACUUM_THRESHOLD):
                    status = spin.main([str(self.poscar), "--incar", str(self.incar), "--no-llm", "--quiet"])
                self.assertEqual(status, 2)
                self.assertTrue(stderr.getvalue().strip())
                if existing:
                    self.assertEqual(self.incar.read_bytes(), original)
                else:
                    self.assertFalse(self.incar.exists())
        self.urlopen.assert_not_called()

    def test_malformed_poscar_returns_two_without_writing_incar(self):
        self.assert_rejected_without_write("invalid POSCAR\n1.0\n")

    def test_incompatible_afm_returns_two_without_writing_incar(self):
        pos = nio_cell()
        self.assert_rejected_without_write(poscar_text(
            lattice=pos.lattice, coords=pos.frac,
            species=pos.species, counts=pos.counts))


if __name__ == "__main__":
    unittest.main()
