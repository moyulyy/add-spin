"""Geometry regressions for chemical motifs used to select magnetic seeds."""

import unittest

import numpy as np

from test_add_spin import poscar_text, spin


def classify(lattice, coords, species, counts, system_type="bulk"):
    pos = spin.parse_poscar_text(poscar_text(
        lattice=lattice, coords=coords, species=species, counts=counts))
    neigh = spin.coordination_analysis(pos)
    return pos, neigh, spin.detect_motif(pos, neigh, system_type)


class MotifGeometryTests(unittest.TestCase):
    def test_bcc_first_shell_contains_eight_neighbors(self):
        lattice = 1.435 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        _, neigh, motif = classify(lattice, [[0, 0, 0]], ["Fe"], [1])
        self.assertEqual(len(neigh[0]), 8)
        self.assertEqual(motif["metal_lattice"], "bcc")

    def test_diamond_carbon_is_not_a_metal(self):
        lattice = 1.7835 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        _, neigh, motif = classify(lattice, [[0, 0, 0], [.25, .25, .25]], ["C"], [2])
        self.assertEqual([len(x) for x in neigh], [4, 4])
        self.assertEqual(motif["family"], "elemental")

    def test_molecular_oxygen_is_not_a_bulk_metal(self):
        _, _, motif = classify(np.eye(3) * 20, [[.5, .5, .47], [.5, .5, .53]],
                               ["O"], [2], "mole")
        self.assertEqual(motif["family"], "molecule")

    def test_cuprite_stoichiometry_is_not_rocksalt(self):
        coords = [[.25, .25, .25], [.25, .75, .75], [.75, .25, .75],
                  [.75, .75, .25], [0, 0, 0], [.5, .5, .5]]
        _, neigh, motif = classify(np.eye(3) * 4.27, coords, ["Cu", "O"], [4, 2])
        self.assertEqual([len(x) for x in neigh[:4]], [2] * 4)
        self.assertEqual(motif["family"], "oxide/compound")

    def test_square_planar_four_coordination_is_not_tetrahedral(self):
        _, neigh, motif = classify(np.diag([4., 4., 20.]),
            [[0, 0, .5], [.5, .5, .5], [.5, 0, .5], [0, .5, .5]],
            ["Ni", "O"], [2, 2], "slab")
        self.assertEqual([len(x) for x in neigh], [4] * 4)
        self.assertEqual(motif["family"], "oxide/compound")

    def test_octahedral_rocksalt_is_retained(self):
        fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        coords = np.concatenate([fcc, (fcc + [.5, 0, 0]) % 1])
        _, neigh, motif = classify(np.eye(3) * 4.17, coords, ["Ni", "O"], [4, 4])
        self.assertEqual([len(x) for x in neigh], [6] * 8)
        self.assertEqual(motif["family"], "rocksalt")
        self.assertIn("候选", motif["name"])

    def test_cubic_perovskite_is_retained(self):
        coords = [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]
        _, _, motif = classify(np.eye(3) * 3.93, coords, ["La", "Fe", "O"], [1, 1, 3])
        self.assertEqual(motif["family"], "perovskite")
        self.assertEqual(motif["a_site"], ["La"])
        self.assertEqual(motif["b_site"], ["Fe"])

    def test_corundum_is_not_caught_by_abo3_rule(self):
        a, c, z, x = 5.035, 13.75, .3553, .306
        lattice = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
        centers = np.array([[0, 0, 0], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]])
        metal = np.array([[0, 0, z], [0, 0, -z], [0, 0, .5 + z], [0, 0, .5 - z]])
        oxygen = np.array([[x, 0, .25], [0, x, .25], [-x, -x, .25],
                           [-x, 0, .75], [0, -x, .75], [x, x, .75]])
        coords = np.concatenate([np.concatenate([(metal + q) % 1 for q in centers]),
                                 np.concatenate([(oxygen + q) % 1 for q in centers])])
        _, neigh, motif = classify(lattice, coords, ["Fe", "O"], [12, 18])
        self.assertEqual([len(x) for x in neigh[:12]], [6] * 12)
        self.assertEqual(motif["family"], "corundum")

    def test_formula_alone_does_not_establish_spinel(self):
        coords = [[.1, .1, .1], [.3, .1, .1], [.5, .1, .1], [.1, .7, .7],
                  [.3, .7, .7], [.5, .7, .7], [.7, .7, .7]]
        for system_type in ("bulk", "slab"):
            with self.subTest(system_type=system_type):
                _, _, motif = classify(np.eye(3) * 20, coords, ["Fe", "O"], [3, 4], system_type)
                self.assertEqual(motif["family"], "oxide/compound")


if __name__ == "__main__":
    unittest.main()
