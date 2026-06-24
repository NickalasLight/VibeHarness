import unittest

from vibeharness.config import Config
from vibeharness.toolset import (FilesystemToolset, Toolset, ToolsetCatalog,
                                 default_catalog)


class _StubToolset(Toolset):
    name = "stub"
    description = "stub toolset"

    def create_tools(self, config):
        return []


class ToolsetCatalogTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_default_catalog_has_fs_and_web(self):
        names = default_catalog().names()
        self.assertIn("fs", names)
        self.assertIn("web", names)

    def test_filesystem_toolset_creates_tools(self):
        tools = FilesystemToolset().create_tools(self.config)
        names = {t.name for t in tools}
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)

    def test_select_unknown_raises(self):
        with self.assertRaises(KeyError):
            default_catalog().select(["fs", "nope"])

    def test_duplicate_toolset_names_rejected(self):
        with self.assertRaises(ValueError):
            ToolsetCatalog([_StubToolset(), _StubToolset()])

    def test_build_registry_merges_selected_toolsets_plus_core(self):
        # The toolset-specific behaviour under test is the MERGE: fs + web tools
        # plus the always-injected `validate` core tool end up in one registry.
        # (The action-schema SHAPE that results is owned by test_registry_schema.)
        catalog = default_catalog()
        registry = catalog.build_registry(catalog.select(["fs", "web"]), self.config)
        names = set(registry.names())
        self.assertIn("read_file", names)   # from fs
        self.assertIn("browse", names)      # from web
        self.assertIn("validate", names)    # core, injected into every registry


if __name__ == "__main__":
    unittest.main()
