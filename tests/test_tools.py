import os
import tempfile
import unittest

from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import (CreateFileTool, ManagePathTool, ReadFileTool,
                                  ReadTracker, WriteFileTool,
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

    def test_write_observation_verb_per_mode(self):
        # The wrapper layer's value is the observation VERB it adds per write mode.
        # (That bytes actually land on disk is FileSystem's spec, in
        # test_filesystem.py.)
        cases = [
            (None, "you wrote the file"),     # default overwrite
            ("append", "appended to"),
            ("prepend", "prepended to"),
        ]
        for mode, verb in cases:
            with self.subTest(mode=mode):
                self._create(self.p("a.txt"), "seed")
                self._reader().run({"path": self.p("a.txt")})  # satisfy overwrite guard
                args = {"path": self.p("a.txt"), "content": "x"}
                if mode:
                    args["mode"] = mode
                res = self._writer().run(args)
                self.assertTrue(res.ok)
                self.assertIn(verb, res.observation)
                ManagePathTool(self.fs).run({"action": "delete", "path": self.p("a.txt")})

    def test_read_returns_file_content_in_observation(self):
        # Wrapper-specific: ReadFileTool surfaces the file content in its observation.
        self._create(self.p("a.txt"), "hi there")
        r = self._reader().run({"path": self.p("a.txt")})
        self.assertTrue(r.ok)
        self.assertIn("hi there", r.observation)

    def test_read_missing_is_error(self):
        res = self._reader().run({"path": self.p("nope.txt")})
        self.assertFalse(res.ok)
        self.assertIn("error", res.observation)

    # NOTE: deep behaviour of list/search/copy/move/delete is owned by
    # test_filesystem.py. Here we only keep the wrapper-specific guards that have
    # no FileSystem equivalent (missing-destination errors on manage_path).
    def test_manage_move_without_destination_fails(self):
        res = ManagePathTool(self.fs).run({"action": "move", "path": self.p("a.txt")})
        self.assertFalse(res.ok)

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

    def test_non_destructive_modes_bypass_read_guard(self):
        # append and prepend add content without discarding the rest, so they
        # bypass the read-before-overwrite guard even on an unread file.
        cases = [("append", "AB"), ("prepend", "BA")]
        for mode, expected in cases:
            with self.subTest(mode=mode):
                self._create(self.p("a.txt"), "A")
                res = self._writer().run(
                    {"path": self.p("a.txt"), "content": "B", "mode": mode})
                self.assertTrue(res.ok)
                self.assertEqual(self.fs.read(self.p("a.txt")), expected)
                ManagePathTool(self.fs).run({"action": "delete", "path": self.p("a.txt")})

    def test_read_tracker_marks_and_reports(self):
        tracker = ReadTracker()
        self.assertFalse(tracker.has_read(self.p("a.txt")))
        tracker.mark(self.p("a.txt"))
        self.assertTrue(tracker.has_read(self.p("a.txt")))
        self.assertFalse(tracker.has_read(self.p("b.txt")))

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
