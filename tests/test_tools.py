import os
import tempfile
import unittest

from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import (CreateFileTool, ListDirectoryTool,
                                  ManagePathTool, ReadFileTool, ReadTracker,
                                  SearchTool, WriteFileTool,
                                  build_default_tools)


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fs = FileSystem()
        self.tracker = ReadTracker()

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    def _writer(self):
        return WriteFileTool(self.fs, self.tracker)

    def _reader(self, limit=1000):
        return ReadFileTool(self.fs, limit, self.tracker)

    def _create(self, path, content):
        return CreateFileTool(self.fs).run({"path": path, "content": content})

    def test_write_then_read_observations(self):
        self._create(self.p("a.txt"), "old")
        self._reader().run({"path": self.p("a.txt")})
        w = self._writer().run({"path": self.p("a.txt"), "content": "hi there"})
        self.assertTrue(w.ok)
        self.assertIn("you wrote the file", w.observation)

        r = self._reader().run({"path": self.p("a.txt")})
        self.assertTrue(r.ok)
        self.assertIn("hi there", r.observation)

    def test_write_append_observation_verb(self):
        self._create(self.p("a.txt"), "a")
        res = self._writer().run({"path": self.p("a.txt"), "content": "b", "mode": "append"})
        self.assertIn("appended to", res.observation)

    def test_read_missing_is_error(self):
        res = self._reader().run({"path": self.p("nope.txt")})
        self.assertFalse(res.ok)
        self.assertIn("error", res.observation)

    def test_list_directory(self):
        self._create(self.p("a.txt"), "x")
        res = ListDirectoryTool(self.fs, 1000).run({"path": self.dir})
        self.assertTrue(res.ok)
        self.assertIn("a.txt", res.observation)

    def test_search_tool(self):
        self._create(self.p("a.txt"), "find ME")
        res = SearchTool(self.fs, 1000).run({"query": "me", "path": self.dir})
        self.assertTrue(res.ok)
        self.assertIn("a.txt", res.observation)

    def test_manage_make_delete_move(self):
        mp = ManagePathTool(self.fs)
        self.assertTrue(mp.run({"action": "make_directory", "path": self.p("d")}).ok)
        self._create(self.p("d", "a.txt"), "x")
        moved = mp.run({"action": "move", "path": self.p("d", "a.txt"),
                        "destination": self.p("b.txt")})
        self.assertTrue(moved.ok)
        self.assertIn("moved", moved.observation)
        self.assertTrue(mp.run({"action": "delete", "path": self.p("b.txt")}).ok)

    def test_manage_move_without_destination_fails(self):
        res = ManagePathTool(self.fs).run({"action": "move", "path": self.p("a.txt")})
        self.assertFalse(res.ok)

    def test_manage_copy(self):
        mp = ManagePathTool(self.fs)
        self._create(self.p("a.txt"), "data")
        res = mp.run({"action": "copy", "path": self.p("a.txt"),
                      "destination": self.p("b.txt")})
        self.assertTrue(res.ok)
        self.assertIn("copied", res.observation)
        self.assertEqual(self.fs.read(self.p("a.txt")), "data")
        self.assertEqual(self.fs.read(self.p("b.txt")), "data")

    def test_manage_copy_without_destination_fails(self):
        res = ManagePathTool(self.fs).run({"action": "copy", "path": self.p("a.txt")})
        self.assertFalse(res.ok)

    # ---- read paging ----
    def test_read_reports_pages(self):
        self._create(self.p("big.txt"), "x" * 25000)
        r = self._reader(limit=20000)
        first = r.run({"path": self.p("big.txt"), "page": 1})
        self.assertTrue(first.ok)
        self.assertIn("page 1 of 3", first.observation)
        self.assertIn("25000 chars total", first.observation)
        last = r.run({"path": self.p("big.txt"), "page": 3})
        self.assertIn("page 3 of 3", last.observation)

    def test_read_out_of_range_page_errors(self):
        self._create(self.p("a.txt"), "small")
        res = self._reader().run({"path": self.p("a.txt"), "page": 2})
        self.assertFalse(res.ok)
        self.assertIn("out of range", res.observation)

    # ---- create_file ----
    def test_create_file_new_ok(self):
        res = self._create(self.p("new.txt"), "hello")
        self.assertTrue(res.ok)
        self.assertIn("you created the file", res.observation)
        self.assertEqual(self.fs.read(self.p("new.txt")), "hello")

    def test_create_file_existing_errors(self):
        self._create(self.p("a.txt"), "x")
        res = self._create(self.p("a.txt"), "y")
        self.assertFalse(res.ok)
        self.assertIn("write_file", res.observation)

    # ---- write_file guards ----
    def test_write_missing_errors_directs_to_create(self):
        res = self._writer().run({"path": self.p("nope.txt"), "content": "x"})
        self.assertFalse(res.ok)
        self.assertIn("create_file", res.observation)

    def test_overwrite_unread_refused(self):
        self._create(self.p("a.txt"), "important")
        res = self._writer().run({"path": self.p("a.txt"), "content": "wiped"})
        self.assertFalse(res.ok)
        self.assertIn("read it first", res.observation)
        self.assertEqual(self.fs.read(self.p("a.txt")), "important")

    def test_overwrite_after_read_ok(self):
        self._create(self.p("a.txt"), "important")
        self._reader().run({"path": self.p("a.txt")})
        res = self._writer().run({"path": self.p("a.txt"), "content": "new"})
        self.assertTrue(res.ok)
        self.assertEqual(self.fs.read(self.p("a.txt")), "new")

    def test_append_without_read_ok(self):
        self._create(self.p("a.txt"), "A")
        res = self._writer().run({"path": self.p("a.txt"), "content": "B", "mode": "append"})
        self.assertTrue(res.ok)
        self.assertEqual(self.fs.read(self.p("a.txt")), "AB")

    def test_default_toolset_names(self):
        names = {t.name for t in build_default_tools(self.fs, 1000)}
        self.assertEqual(names, {"list_directory", "read_file", "create_file",
                                 "write_file", "search", "manage_path"})

    def test_shared_tracker_across_read_and_write(self):
        tools = {t.name: t for t in build_default_tools(self.fs, 20000)}
        self._create(self.p("a.txt"), "orig")
        tools["read_file"].run({"path": self.p("a.txt")})
        res = tools["write_file"].run({"path": self.p("a.txt"), "content": "new"})
        self.assertTrue(res.ok)


if __name__ == "__main__":
    unittest.main()
