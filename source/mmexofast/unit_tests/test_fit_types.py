"""Unit tests for fit_types module."""

import unittest

from mmexofast import fit_types


# ---------------------------------------------------------------------------
# Module-level fixtures shared across test classes
# ---------------------------------------------------------------------------

#: All eight recognised binary_model_type values.
VALID_BINARY_MODEL_TYPES = [
    'Wide', 'Close', 'CloseUpper', 'CloseLower',
    'Wide_alt', 'Close_alt', 'CloseUpper_alt', 'CloseLower_alt',
]

#: Representative sample of strings that must be rejected.
INVALID_BINARY_MODEL_TYPES_SAMPLES = [
    'wide',              # wrong capitalisation
    'close',             # wrong capitalisation
    'closeUpper',        # wrong capitalisation
    'closeLower',        # wrong capitalisation
    'WIDE',              # all-caps
    'CLOSE',             # all-caps
    'resonant',          # unrecognised
    'central',           # unrecognised
    'planetary',         # unrecognised
    'Wide_alt_alt',      # double-alt suffix
    'wide_alt',          # wrong capitalisation on base
    'CloseUpper_Alt',    # wrong capitalisation on 'Alt'
]


# ============================================================================
# Enum tests
# ============================================================================


class TestEnums(unittest.TestCase):
    """Test enum definitions."""

    def test_lens_type_values(self):
        expected_values = ['POINT', 'BINARY']
        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.LensType, value))

    def test_source_type_values(self):
        expected_values = ['POINT', 'FINITE']
        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.SourceType, value))

    def test_parallax_branch_values(self):
        expected_values = ['NONE', 'U0_PLUS', 'U0_MINUS', 'U0_PP', 'U0_PM', 'U0_MP', 'U0_MM']
        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.ParallaxBranch, value))

    def test_lens_orb_motion_values(self):
        expected_values = ['NONE', 'KEPLER', 'ORB_2D']
        for value in expected_values:
            with self.subTest(value=value):
                self.assertTrue(hasattr(fit_types.LensOrbMotion, value))


# ============================================================================
# Tag-constant tests
# ============================================================================


class TestTagConstants(unittest.TestCase):
    """Test TAG constant dictionaries."""

    def test_lens_tags_exist(self):
        self.assertIsInstance(fit_types.LENS_TAGS, dict)
        self.assertGreater(len(fit_types.LENS_TAGS), 0)
        for tag, lens_type in fit_types.LENS_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(lens_type, fit_types.LensType)

    def test_source_tags_exist(self):
        self.assertIsInstance(fit_types.SOURCE_TAGS, dict)
        self.assertGreater(len(fit_types.SOURCE_TAGS), 0)
        for tag, source_type in fit_types.SOURCE_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(source_type, fit_types.SourceType)

    def test_parallax_branch_tags_exist(self):
        self.assertIsInstance(fit_types.PARALLAX_BRANCH_TAGS, dict)
        self.assertGreater(len(fit_types.PARALLAX_BRANCH_TAGS), 0)
        for tag, branch in fit_types.PARALLAX_BRANCH_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(branch, fit_types.ParallaxBranch)

    def test_lens_motion_tags_exist(self):
        self.assertIsInstance(fit_types.LENS_MOTION_TAGS, dict)
        self.assertGreater(len(fit_types.LENS_MOTION_TAGS), 0)
        for tag, motion in fit_types.LENS_MOTION_TAGS.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(motion, fit_types.LensOrbMotion)

    # ------------------------------------------------------------------
    # UPDATED: BINARY_MODEL_TYPES constant
    # ------------------------------------------------------------------

    def test_binary_model_types_constant_exists(self):
        """fit_types must export a BINARY_MODEL_TYPES collection."""
        self.assertTrue(
            hasattr(fit_types, 'BINARY_MODEL_TYPES'),
            "fit_types.BINARY_MODEL_TYPES not found",
        )
        self.assertIsInstance(
            fit_types.BINARY_MODEL_TYPES,
            (list, tuple, set, frozenset),
        )

    def test_binary_model_types_contains_exactly_valid_values(self):
        """BINARY_MODEL_TYPES must equal the canonical set of eight values."""
        actual = set(fit_types.BINARY_MODEL_TYPES)
        expected = set(VALID_BINARY_MODEL_TYPES)
        self.assertEqual(actual, expected)

    def test_binary_model_types_base_values_present(self):
        """Each of the four base types must be in BINARY_MODEL_TYPES."""
        for base in ('Wide', 'Close', 'CloseUpper', 'CloseLower'):
            with self.subTest(base=base):
                self.assertIn(base, fit_types.BINARY_MODEL_TYPES)

    def test_binary_model_types_alt_values_present(self):
        """Each base type must have a corresponding _alt variant."""
        for base in ('Wide', 'Close', 'CloseUpper', 'CloseLower'):
            with self.subTest(alt=f'{base}_alt'):
                self.assertIn(f'{base}_alt', fit_types.BINARY_MODEL_TYPES)


# ============================================================================
# FitKey dataclass tests
# ============================================================================


class TestFitKey(unittest.TestCase):
    """Test FitKey dataclass."""

    def test_creation_minimal(self):
        """FitKey can be created with only the four required fields."""
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)
        # UPDATED: new field must default to None
        self.assertIsNone(key.binary_model_type)

    def test_creation_with_locations(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used='ground+Spitzer',
        )
        self.assertEqual(key.locations_used, 'ground+Spitzer')

    def test_equality_same_values(self):
        key1 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        key2 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        self.assertEqual(key1, key2)

    def test_equality_different_values(self):
        key1 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        key2 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        self.assertNotEqual(key1, key2)

    def test_hashability(self):
        key1 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        key2 = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        test_dict = {key1: 'value1'}
        self.assertEqual(test_dict[key2], 'value1')

    def test_string_representation(self):
        """FitKey repr contains the values of all set fields."""
        test_cases = [
            (
                {
                    'lens_type': fit_types.LensType.POINT,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.NONE,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE,
                },
                ['POINT', 'NONE'],
            ),
            (
                {
                    'lens_type': fit_types.LensType.POINT,
                    'source_type': fit_types.SourceType.FINITE,
                    'parallax_branch': fit_types.ParallaxBranch.U0_PLUS,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE,
                },
                ['POINT', 'FINITE', 'U0_PLUS'],
            ),
            (
                {
                    'lens_type': fit_types.LensType.BINARY,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.U0_MM,
                    'lens_orb_motion': fit_types.LensOrbMotion.KEPLER,
                },
                ['BINARY', 'U0_MM', 'KEPLER'],
            ),
            (
                {
                    'lens_type': fit_types.LensType.POINT,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.NONE,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE,
                    'locations_used': 'ground+Spitzer',
                },
                ['POINT', 'NONE', 'ground+Spitzer'],
            ),
            # UPDATED: use 'Wide' (canonical capitalisation)
            (
                {
                    'lens_type': fit_types.LensType.BINARY,
                    'source_type': fit_types.SourceType.POINT,
                    'parallax_branch': fit_types.ParallaxBranch.NONE,
                    'lens_orb_motion': fit_types.LensOrbMotion.NONE,
                    'binary_model_type': 'Wide',
                },
                ['BINARY', 'Wide'],
            ),
        ]

        for params, expected_substrings in test_cases:
            with self.subTest(params=params):
                key = fit_types.FitKey(**params)
                key_str = str(key)
                self.assertIsInstance(key_str, str)
                for substring in expected_substrings:
                    self.assertIn(substring, key_str)


# ============================================================================
# binary_model_type field tests
# ============================================================================


class TestFitKeyBinaryModelType(unittest.TestCase):
    """Tests for the optional binary_model_type field on FitKey.

    Contract
    --------
    * Defaults to ``None`` for every ``FitKey``.
    * Accepted values are exactly the eight strings in
      ``VALID_BINARY_MODEL_TYPES`` (four base names + their ``_alt`` variants).
    * A non-``None`` value is only permitted when
      ``lens_type == LensType.BINARY``; any other combination raises
      ``ValueError``.
    * An empty string raises ``ValueError``.
    * Strings not in the accepted set raise ``ValueError``.
    * Reserved parser keywords raise ``ValueError``.
    * The field participates fully in equality and hashing.
    """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _binary_key(self, **kwargs):
        defaults = dict(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        defaults.update(kwargs)
        return fit_types.FitKey(**defaults)

    # ------------------------------------------------------------------
    # Default value
    # ------------------------------------------------------------------

    def test_defaults_to_none_for_point_lens(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        self.assertIsNone(key.binary_model_type)

    def test_defaults_to_none_for_binary_lens(self):
        key = self._binary_key()
        self.assertIsNone(key.binary_model_type)

    # ------------------------------------------------------------------
    # Valid construction
    # ------------------------------------------------------------------

    def test_can_be_set_to_any_valid_value(self):
        """Every value in VALID_BINARY_MODEL_TYPES must be accepted."""
        for model_type in VALID_BINARY_MODEL_TYPES:
            with self.subTest(binary_model_type=model_type):
                key = self._binary_key(binary_model_type=model_type)
                self.assertEqual(key.binary_model_type, model_type)

    def test_stored_as_str(self):
        key = self._binary_key(binary_model_type='Wide')
        self.assertIsInstance(key.binary_model_type, str)

    def test_explicit_none_always_valid(self):
        """Explicit binary_model_type=None is valid for every LensType."""
        for lens_type in fit_types.LensType:
            with self.subTest(lens_type=lens_type):
                key = fit_types.FitKey(
                    lens_type=lens_type,
                    source_type=fit_types.SourceType.POINT,
                    parallax_branch=fit_types.ParallaxBranch.NONE,
                    lens_orb_motion=fit_types.LensOrbMotion.NONE,
                    binary_model_type=None,
                )
                self.assertIsNone(key.binary_model_type)

    def test_alt_suffix_is_accepted(self):
        """Each '_alt' variant is independently valid."""
        for base in ('Wide', 'Close', 'CloseUpper', 'CloseLower'):
            with self.subTest(alt=f'{base}_alt'):
                key = self._binary_key(binary_model_type=f'{base}_alt')
                self.assertEqual(key.binary_model_type, f'{base}_alt')

    # ------------------------------------------------------------------
    # Validation errors — lens-type constraint
    # ------------------------------------------------------------------

    def test_raises_for_point_lens_with_non_none_value(self):
        """Setting binary_model_type on LensType.POINT must raise ValueError."""
        with self.assertRaises(ValueError):
            fit_types.FitKey(
                lens_type=fit_types.LensType.POINT,
                source_type=fit_types.SourceType.POINT,
                parallax_branch=fit_types.ParallaxBranch.NONE,
                lens_orb_motion=fit_types.LensOrbMotion.NONE,
                binary_model_type='Wide',
            )

    def test_raises_for_fspl_with_non_none_value(self):
        """FSPL (point lens, finite source) must also reject a non-None value."""
        with self.assertRaises(ValueError):
            fit_types.FitKey(
                lens_type=fit_types.LensType.POINT,
                source_type=fit_types.SourceType.FINITE,
                parallax_branch=fit_types.ParallaxBranch.NONE,
                lens_orb_motion=fit_types.LensOrbMotion.NONE,
                binary_model_type='Close',
            )

    # ------------------------------------------------------------------
    # Validation errors — value constraint
    # ------------------------------------------------------------------

    def test_raises_for_empty_string(self):
        with self.assertRaises(ValueError):
            self._binary_key(binary_model_type='')

    def test_raises_for_invalid_binary_model_type_string(self):
        """Every value in INVALID_BINARY_MODEL_TYPES_SAMPLES must raise ValueError."""
        for bad_value in INVALID_BINARY_MODEL_TYPES_SAMPLES:
            with self.subTest(binary_model_type=bad_value):
                with self.assertRaises(ValueError):
                    self._binary_key(binary_model_type=bad_value)

    def test_raises_for_reserved_keyword_static(self):
        with self.assertRaises(ValueError):
            self._binary_key(binary_model_type='static')

    def test_raises_for_reserved_keyword_par(self):
        with self.assertRaises(ValueError):
            self._binary_key(binary_model_type='par')

    def test_raises_for_reserved_motion_keywords(self):
        for reserved in ['2Dorb', 'kep', 'none']:
            with self.subTest(reserved=reserved):
                with self.assertRaises(ValueError):
                    self._binary_key(binary_model_type=reserved)

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def test_equal_when_same_binary_model_type(self):
        self.assertEqual(
            self._binary_key(binary_model_type='Wide'),
            self._binary_key(binary_model_type='Wide'),
        )

    def test_not_equal_when_different_binary_model_type(self):
        key_wide = self._binary_key(binary_model_type='Wide')
        key_close = self._binary_key(binary_model_type='Close')
        key_none = self._binary_key(binary_model_type=None)

        self.assertNotEqual(key_wide, key_close)
        self.assertNotEqual(key_wide, key_none)
        self.assertNotEqual(key_close, key_none)

    def test_base_and_alt_not_equal(self):
        """'Wide' and 'Wide_alt' are distinct values and must not compare equal."""
        self.assertNotEqual(
            self._binary_key(binary_model_type='Wide'),
            self._binary_key(binary_model_type='Wide_alt'),
        )

    def test_none_and_unset_are_equal(self):
        self.assertEqual(
            self._binary_key(),
            self._binary_key(binary_model_type=None),
        )

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def test_hashable_with_binary_model_type_set(self):
        key1 = self._binary_key(binary_model_type='Wide')
        key2 = self._binary_key(binary_model_type='Wide')
        self.assertEqual({key1: 'found'}[key2], 'found')

    def test_different_binary_model_type_different_hash(self):
        self.assertNotEqual(
            hash(self._binary_key(binary_model_type='Wide')),
            hash(self._binary_key(binary_model_type='Close')),
        )

    def test_base_and_alt_have_different_hash(self):
        self.assertNotEqual(
            hash(self._binary_key(binary_model_type='Wide')),
            hash(self._binary_key(binary_model_type='Wide_alt')),
        )

    def test_none_and_set_have_different_hash(self):
        self.assertNotEqual(
            hash(self._binary_key(binary_model_type=None)),
            hash(self._binary_key(binary_model_type='Wide')),
        )

    # ------------------------------------------------------------------
    # Combinations with other optional fields
    # ------------------------------------------------------------------

    def test_combined_with_parallax(self):
        key = self._binary_key(
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            binary_model_type='Wide',
        )
        self.assertEqual(key.binary_model_type, 'Wide')
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_PLUS)

    def test_combined_with_orbital_motion_2d(self):
        key = self._binary_key(
            lens_orb_motion=fit_types.LensOrbMotion.ORB_2D,
            binary_model_type='Close',
        )
        self.assertEqual(key.binary_model_type, 'Close')
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.ORB_2D)

    def test_combined_with_kepler_and_parallax(self):
        key = self._binary_key(
            parallax_branch=fit_types.ParallaxBranch.U0_MM,
            lens_orb_motion=fit_types.LensOrbMotion.KEPLER,
            binary_model_type='Wide',
        )
        self.assertEqual(key.binary_model_type, 'Wide')
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_MM)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.KEPLER)

    def test_combined_with_locations_used(self):
        key = self._binary_key(
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            binary_model_type='Wide',
            locations_used='ground+Spitzer',
        )
        self.assertEqual(key.binary_model_type, 'Wide')
        self.assertEqual(key.locations_used, 'ground+Spitzer')

    def test_combined_alt_with_kepler_and_parallax(self):
        key = self._binary_key(
            parallax_branch=fit_types.ParallaxBranch.U0_PP,
            lens_orb_motion=fit_types.LensOrbMotion.KEPLER,
            binary_model_type='Close_alt',
        )
        self.assertEqual(key.binary_model_type, 'Close_alt')

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def test_repr_contains_binary_model_type_value(self):
        key = self._binary_key(binary_model_type='Wide')
        self.assertIn('Wide', repr(key))

    def test_repr_contains_none_when_unset(self):
        key = self._binary_key()
        self.assertIn('None', repr(key))


# ============================================================================
# label <-> FitKey conversion — binary_model_type
# ============================================================================


class TestLabelConversionsBinaryModelType(unittest.TestCase):
    """Label parsing/generation for binary_model_type.

    Grammar extension
    -----------------
    The optional token sits **immediately after** the base tag, before any
    motion or parallax suffix:

        "<base> <binary_model_type> static"
        "<base> <binary_model_type> par <branch>"
        "<base> <binary_model_type> <motion>"
        "<base> <binary_model_type> <motion> par <branch>"

    Omitting the token leaves binary_model_type=None (backwards-compatible).
    """

    # ------------------------------------------------------------------
    # Parsing: binary_model_type present
    # ------------------------------------------------------------------

    def test_parse_wide_static(self):
        key = fit_types.label_to_model_key('2L1S Wide static')
        self.assertEqual(key.lens_type, fit_types.LensType.BINARY)
        self.assertEqual(key.binary_model_type, 'Wide')
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)

    def test_parse_close_with_parallax(self):
        key = fit_types.label_to_model_key('2L1S Close par u0+')
        self.assertEqual(key.binary_model_type, 'Close')
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_PLUS)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)

    def test_parse_wide_with_2d_orbital_motion(self):
        key = fit_types.label_to_model_key('2L1S Wide 2Dorb')
        self.assertEqual(key.binary_model_type, 'Wide')
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.ORB_2D)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)

    def test_parse_close_with_kepler_motion(self):
        key = fit_types.label_to_model_key('2L1S Close kep')
        self.assertEqual(key.binary_model_type, 'Close')
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.KEPLER)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)

    def test_parse_wide_with_2d_orbital_motion_and_parallax(self):
        key = fit_types.label_to_model_key('2L1S Wide 2Dorb par u0+')
        self.assertEqual(key.binary_model_type, 'Wide')
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.ORB_2D)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_PLUS)

    def test_parse_close_with_kepler_and_parallax(self):
        key = fit_types.label_to_model_key('2L1S Close kep par u0--')
        self.assertEqual(key.binary_model_type, 'Close')
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.KEPLER)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_MM)

    def test_parse_all_valid_binary_model_types(self):
        """Every value in VALID_BINARY_MODEL_TYPES parses successfully."""
        for model_type in VALID_BINARY_MODEL_TYPES:
            with self.subTest(binary_model_type=model_type):
                label = f'2L1S {model_type} static'
                key = fit_types.label_to_model_key(label)
                self.assertEqual(key.binary_model_type, model_type)

    def test_parse_closeupper_and_closelower(self):
        """Compound names CloseUpper and CloseLower parse correctly."""
        for model_type in ('CloseUpper', 'CloseLower'):
            with self.subTest(binary_model_type=model_type):
                key = fit_types.label_to_model_key(f'2L1S {model_type} static')
                self.assertEqual(key.binary_model_type, model_type)

    def test_parse_alt_variants(self):
        """_alt variants parse to the exact string including the suffix."""
        for base in ('Wide', 'Close', 'CloseUpper', 'CloseLower'):
            alt = f'{base}_alt'
            with self.subTest(binary_model_type=alt):
                key = fit_types.label_to_model_key(f'2L1S {alt} static')
                self.assertEqual(key.binary_model_type, alt)

    # ------------------------------------------------------------------
    # Parsing: backwards-compatibility (no binary_model_type token)
    # ------------------------------------------------------------------

    def test_existing_binary_labels_give_none_binary_model_type(self):
        for label in (
            '2L1S static',
            '2L1S par u0+',
            '2L1S 2Dorb',
            '2L1S kep',
            '2L1S 2Dorb par u0+',
            '2L1S kep par u0--',
        ):
            with self.subTest(label=label):
                self.assertIsNone(
                    fit_types.label_to_model_key(label).binary_model_type
                )

    def test_pspl_fspl_labels_give_none_binary_model_type(self):
        for label in ('PSPL static', 'FSPL static', 'PSPL par u0+'):
            with self.subTest(label=label):
                self.assertIsNone(
                    fit_types.label_to_model_key(label).binary_model_type
                )

    # ------------------------------------------------------------------
    # Parsing: invalid binary_model_type tokens
    # ------------------------------------------------------------------

    def test_parse_invalid_binary_model_type_raises(self):
        """Unrecognised binary_model_type token in a label must raise an error."""
        for bad in ('wide', 'close', 'WIDE', 'resonant', 'Wide_alt_alt'):
            with self.subTest(bad_token=bad):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(f'2L1S {bad} static')

    # ------------------------------------------------------------------
    # Generation: binary_model_type present
    # ------------------------------------------------------------------

    def test_generate_wide_static(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            binary_model_type='Wide',
        )
        self.assertEqual(fit_types.model_key_to_label(key), '2L1S Wide static')

    def test_generate_close_with_parallax(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            binary_model_type='Close',
        )
        self.assertEqual(fit_types.model_key_to_label(key), '2L1S Close par u0+')

    def test_generate_wide_with_2d_motion(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.ORB_2D,
            binary_model_type='Wide',
        )
        self.assertEqual(fit_types.model_key_to_label(key), '2L1S Wide 2Dorb')

    def test_generate_wide_with_kepler_and_parallax(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_MM,
            lens_orb_motion=fit_types.LensOrbMotion.KEPLER,
            binary_model_type='Wide',
        )
        self.assertEqual(fit_types.model_key_to_label(key), '2L1S Wide kep par u0--')

    def test_generate_alt_variant_label(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            binary_model_type='Wide_alt',
        )
        self.assertEqual(fit_types.model_key_to_label(key), '2L1S Wide_alt static')

    def test_generate_closeupper_with_2d_motion_and_parallax(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.ORB_2D,
            binary_model_type='CloseUpper',
        )
        self.assertEqual(
            fit_types.model_key_to_label(key), '2L1S CloseUpper 2Dorb par u0+'
        )

    # ------------------------------------------------------------------
    # Generation: binary_model_type absent → existing behaviour unchanged
    # ------------------------------------------------------------------

    def test_generate_label_without_binary_model_type_unchanged(self):
        cases = [
            (
                fit_types.FitKey(
                    lens_type=fit_types.LensType.BINARY,
                    source_type=fit_types.SourceType.POINT,
                    parallax_branch=fit_types.ParallaxBranch.NONE,
                    lens_orb_motion=fit_types.LensOrbMotion.NONE,
                ),
                '2L1S static',
            ),
            (
                fit_types.FitKey(
                    lens_type=fit_types.LensType.BINARY,
                    source_type=fit_types.SourceType.POINT,
                    parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
                    lens_orb_motion=fit_types.LensOrbMotion.NONE,
                ),
                '2L1S par u0+',
            ),
            (
                fit_types.FitKey(
                    lens_type=fit_types.LensType.BINARY,
                    source_type=fit_types.SourceType.POINT,
                    parallax_branch=fit_types.ParallaxBranch.NONE,
                    lens_orb_motion=fit_types.LensOrbMotion.ORB_2D,
                ),
                '2L1S 2Dorb',
            ),
        ]
        for key, expected_label in cases:
            with self.subTest(expected_label=expected_label):
                self.assertEqual(fit_types.model_key_to_label(key), expected_label)

    # ------------------------------------------------------------------
    # Token position
    # ------------------------------------------------------------------

    def test_binary_model_type_token_position_in_label(self):
        """binary_model_type token must be the second token (index 1)."""
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            binary_model_type='Wide',
        )
        tokens = fit_types.model_key_to_label(key).split()
        self.assertEqual(tokens[0], '2L1S')
        self.assertEqual(tokens[1], 'Wide')   # immediately after base

    # ------------------------------------------------------------------
    # Round-trips: label → key → label
    # ------------------------------------------------------------------

    def test_round_trip_label_first(self):
        """label → key → label is lossless for every valid binary_model_type."""
        original_labels = [
            '2L1S Wide static',
            '2L1S Close static',
            '2L1S CloseUpper static',
            '2L1S CloseLower static',
            '2L1S Wide_alt static',
            '2L1S Close_alt static',
            '2L1S CloseUpper_alt static',
            '2L1S CloseLower_alt static',
            '2L1S Wide par u0+',
            '2L1S Close par u0-',
            '2L1S Wide 2Dorb',
            '2L1S Close kep',
            '2L1S Wide 2Dorb par u0+',
            '2L1S Close kep par u0--',
            '2L1S CloseUpper par u0++',
            '2L1S CloseLower_alt 2Dorb par u0-+',
        ]
        for label in original_labels:
            with self.subTest(label=label):
                regenerated = fit_types.model_key_to_label(
                    fit_types.label_to_model_key(label)
                )
                self.assertEqual(
                    regenerated,
                    label,
                    msg=f"Round-trip failed: {label!r} → {regenerated!r}",
                )

    def test_round_trip_key_first(self):
        """key → label → key is lossless for every valid binary_model_type."""
        base = dict(
            lens_type=fit_types.LensType.BINARY,
            source_type=fit_types.SourceType.POINT,
        )
        keys = [
            fit_types.FitKey(
                **base,
                parallax_branch=fit_types.ParallaxBranch.NONE,
                lens_orb_motion=fit_types.LensOrbMotion.NONE,
                binary_model_type='Wide',
            ),
            fit_types.FitKey(
                **base,
                parallax_branch=fit_types.ParallaxBranch.U0_PLUS,
                lens_orb_motion=fit_types.LensOrbMotion.NONE,
                binary_model_type='Close',
            ),
            fit_types.FitKey(
                **base,
                parallax_branch=fit_types.ParallaxBranch.NONE,
                lens_orb_motion=fit_types.LensOrbMotion.KEPLER,
                binary_model_type='CloseUpper',
            ),
            fit_types.FitKey(
                **base,
                parallax_branch=fit_types.ParallaxBranch.U0_MM,
                lens_orb_motion=fit_types.LensOrbMotion.ORB_2D,
                binary_model_type='CloseLower',
            ),
            fit_types.FitKey(
                **base,
                parallax_branch=fit_types.ParallaxBranch.NONE,
                lens_orb_motion=fit_types.LensOrbMotion.NONE,
                binary_model_type='Wide_alt',
            ),
            fit_types.FitKey(
                **base,
                parallax_branch=fit_types.ParallaxBranch.U0_PP,
                lens_orb_motion=fit_types.LensOrbMotion.KEPLER,
                binary_model_type='Close_alt',
            ),
        ]
        for key in keys:
            with self.subTest(key=key):
                recovered = fit_types.label_to_model_key(
                    fit_types.model_key_to_label(key)
                )
                self.assertEqual(recovered, key)


# ============================================================================
# Invalid binary_model_type position / context
# ============================================================================


class TestInvalidBinaryModelType(unittest.TestCase):
    """Parser rejects binary_model_type tokens in wrong positions."""

    def test_point_lens_label_with_spurious_token_is_invalid(self):
        """Point-lens bases have no binary_model_type slot; any token there fails."""
        invalid_labels = [
            'PSPL Wide static',    # 'Wide' has no meaning for PSPL
            'FSPL Close par u0+',  # 'Close' has no meaning for FSPL
        ]
        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)

    def test_binary_model_type_after_motion_token_is_invalid(self):
        """binary_model_type must come before, not after, the motion token."""
        invalid_labels = [
            '2L1S 2Dorb Wide',         # 'Wide' after motion token
            '2L1S kep Close',           # 'Close' after motion token
            '2L1S 2Dorb par u0+ Wide',  # trailing valid token after full suffix
        ]
        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)

    def test_binary_model_type_after_parallax_suffix_is_invalid(self):
        """binary_model_type must come before the parallax suffix."""
        invalid_labels = [
            '2L1S par u0+ Wide',  # 'Wide' after complete parallax suffix
            '2L1S static Wide',   # 'Wide' after 'static'
        ]
        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)


# ============================================================================
# Existing test classes (unchanged)
# ============================================================================


class TestLabelConversions(unittest.TestCase):
    """Test label <-> model key conversion functions."""

    def test_label_to_key_pspl_static(self):
        key = fit_types.label_to_model_key('PSPL static')
        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_fspl_static(self):
        key = fit_types.label_to_model_key('FSPL static')
        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.FINITE)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.NONE)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_parallax_u0_plus(self):
        key = fit_types.label_to_model_key('PSPL par u0+')
        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_PLUS)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_parallax_u0_minus(self):
        key = fit_types.label_to_model_key('PSPL par u0-')
        self.assertEqual(key.lens_type, fit_types.LensType.POINT)
        self.assertEqual(key.source_type, fit_types.SourceType.POINT)
        self.assertEqual(key.parallax_branch, fit_types.ParallaxBranch.U0_MINUS)
        self.assertEqual(key.lens_orb_motion, fit_types.LensOrbMotion.NONE)
        self.assertIsNone(key.locations_used)

    def test_label_to_key_parallax_multi_loc(self):
        test_cases = [
            ('PSPL par u0++', fit_types.ParallaxBranch.U0_PP),
            ('PSPL par u0--', fit_types.ParallaxBranch.U0_MM),
            ('PSPL par u0+-', fit_types.ParallaxBranch.U0_PM),
            ('PSPL par u0-+', fit_types.ParallaxBranch.U0_MP),
        ]
        for label, expected_branch in test_cases:
            with self.subTest(label=label):
                self.assertEqual(
                    fit_types.label_to_model_key(label).parallax_branch,
                    expected_branch,
                )

    def test_label_to_key_invalid(self):
        with self.assertRaises((ValueError, KeyError)):
            fit_types.label_to_model_key('InvalidLabel')

    def test_key_to_label_pspl_static(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        self.assertEqual(fit_types.model_key_to_label(key), 'PSPL static')

    def test_key_to_label_with_locations(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used='ground+Spitzer',
        )
        self.assertIn('PSPL', fit_types.model_key_to_label(key))

    def test_round_trip_conversion(self):
        original_labels = [
            'PSPL static',
            'FSPL static',
            'PSPL par u0+',
            'PSPL par u0-',
            'PSPL par u0++',
            '2L1S 2Dorb',
            '2L1S kep par u0+',
        ]
        for label in original_labels:
            with self.subTest(label=label):
                key = fit_types.label_to_model_key(label)
                key2 = fit_types.label_to_model_key(fit_types.model_key_to_label(key))
                self.assertEqual(key, key2)

    def test_all_enum_combinations(self):
        test_cases = [
            (
                fit_types.LensType.POINT, fit_types.SourceType.POINT,
                fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.NONE,
                ['PSPL', 'static'],
            ),
            (
                fit_types.LensType.POINT, fit_types.SourceType.FINITE,
                fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.NONE,
                ['FSPL', 'static'],
            ),
            (
                fit_types.LensType.POINT, fit_types.SourceType.POINT,
                fit_types.ParallaxBranch.U0_PLUS, fit_types.LensOrbMotion.NONE,
                ['PSPL', 'par', 'u0+'],
            ),
            (
                fit_types.LensType.POINT, fit_types.SourceType.FINITE,
                fit_types.ParallaxBranch.U0_MINUS, fit_types.LensOrbMotion.NONE,
                ['FSPL', 'par', 'u0-'],
            ),
            (
                fit_types.LensType.BINARY, fit_types.SourceType.POINT,
                fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.KEPLER,
                ['2L1S', 'kep'],
            ),
            (
                fit_types.LensType.BINARY, fit_types.SourceType.POINT,
                fit_types.ParallaxBranch.NONE, fit_types.LensOrbMotion.ORB_2D,
                ['2L1S', '2Dorb'],
            ),
        ]
        for lens, source, parallax, motion, expected_parts in test_cases:
            with self.subTest(lens=lens, source=source, parallax=parallax, motion=motion):
                key = fit_types.FitKey(
                    lens_type=lens,
                    source_type=source,
                    parallax_branch=parallax,
                    lens_orb_motion=motion,
                )
                label = fit_types.model_key_to_label(key)
                for part in expected_parts:
                    self.assertIn(part, label)

    def test_locations_none_vs_absent(self):
        key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used=None,
        )
        label = fit_types.model_key_to_label(key)
        self.assertIsInstance(label, str)
        self.assertNotIn('(', label)

    def test_label_order_strict(self):
        invalid_orders = [
            'par u0+ PSPL',
            'static FSPL',
            'u0+ par PSPL',
            'kep 2L1S',
            '2Dorb 2L1S',
        ]
        for label in invalid_orders:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)

    def test_label_to_key_binary_motion_no_parallax(self):
        test_cases = [
            ('2L1S 2Dorb', fit_types.LensOrbMotion.ORB_2D, fit_types.ParallaxBranch.NONE),
            ('2L1S kep', fit_types.LensOrbMotion.KEPLER, fit_types.ParallaxBranch.NONE),
        ]
        for label, expected_motion, expected_parallax in test_cases:
            with self.subTest(label=label):
                key = fit_types.label_to_model_key(label)
                self.assertEqual(key.lens_orb_motion, expected_motion)
                self.assertEqual(key.parallax_branch, expected_parallax)


class TestInvalidLabels(unittest.TestCase):
    """Test error handling for invalid labels."""

    def test_empty_label(self):
        with self.assertRaises((ValueError, KeyError)):
            fit_types.label_to_model_key('')

    def test_label_with_nonexistent_keys(self):
        invalid_labels = [
            'InvalidLens static',
            'PSPL invalidkey',
            'PSPL par invalidbranch',
        ]
        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)

    def test_point_lens_with_orbital_motion(self):
        invalid_combinations = ['PSPL kep', 'PSPL 2Dorb', 'FSPL kep']
        for label in invalid_combinations:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)

    def test_binary_motion_with_invalid_token(self):
        invalid_labels = [
            '2L1S 2Dorb invalidToken',
            '2L1S kep badToken',
        ]
        for label in invalid_labels:
            with self.subTest(label=label):
                with self.assertRaises((ValueError, KeyError)):
                    fit_types.label_to_model_key(label)


class TestInvalidKeys(unittest.TestCase):
    """Test that TAGS mappings are complete."""

    def test_tags_completeness(self):
        for lens in fit_types.LensType:
            for source in fit_types.SourceType:
                found = any(
                    fit_types.LENS_TAGS[tag] == lens
                    and fit_types.SOURCE_TAGS[tag] == source
                    for tag in fit_types.LENS_TAGS
                )
                if lens == fit_types.LensType.BINARY and source == fit_types.SourceType.FINITE:
                    continue

                self.assertTrue(found, f"No tag for {lens}+{source}")

        for motion in fit_types.LensOrbMotion:
            self.assertIn(
                motion,
                fit_types.LENS_MOTION_TAGS.values(),
                f"LensOrbMotion.{motion} not in LENS_MOTION_TAGS",
            )

        for branch in fit_types.ParallaxBranch:
            self.assertIn(
                branch,
                fit_types.PARALLAX_BRANCH_TAGS.values(),
                f"ParallaxBranch.{branch} not in PARALLAX_BRANCH_TAGS",
            )


if __name__ == '__main__':
    unittest.main()