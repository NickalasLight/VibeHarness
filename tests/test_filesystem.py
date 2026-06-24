import os
import tempfile
import unittest

from vibeharness.filesystem import FileSystem, FileSystemError


class FileSystemTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fs = FileSystem()

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    # ---- write / read ----
    def test_write_then_read_roundtrip(self):
        n = self.fs.write(self.p("a.txt"), "hello")
        self.assertEqual(n, 5)
        self.assertEqual(self.fs.read(self.p("a.txt")), "hello")

    def test_write_creates_parent_dirs(self):
        self.fs.write(self.p("nested", "deep", "a.txt"), "x")
        self.assertTrue(os.path.isfile(self.p("nested", "deep", "a.txt")))

    def test_write_modes(self):
        path = self.p("a.txt")
        self.fs.write(path, "B")
        self.fs.write(path, "A", mode="prepend")
        self.fs.write(path, "C", mode="append")
        self.assertEqual(self.fs.read(path), "ABC")

    def test_write_unknown_mode_raises(self):
        with self.assertRaises(FileSystemError):
            self.fs.write(self.p("a.txt"), "x", mode="sideways")

    def test_read_missing_raises(self):
        with self.assertRaises(FileSystemError):
            self.fs.read(self.p("nope.txt"))

    def test_read_truncates(self):
        self.fs.write(self.p("a.txt"), "x" * 100)
        out = self.fs.read(self.p("a.txt"), max_chars=10)
        self.assertIn("truncated", out)
        self.assertTrue(out.startswith("x" * 10))

    # ---- listing ----
    def test_list_dir_marks_directories(self):
        self.fs.make_directory(self.p("sub"))
        self.fs.write(self.p("a.txt"), "x")
        entries = self.fs.list_dir(self.dir)
        self.assertIn("sub/", entries)
        self.assertIn("a.txt", entries)

    def test_list_missing_dir_raises(self):
        with self.assertRaises(FileSystemError):
            self.fs.list_dir(self.p("nope"))

    # ---- search ----
    def test_search_content(self):
        self.fs.write(self.p("a.txt"), "alpha\nNEEDLE here\nbeta")
        hits = self.fs.search("needle", self.dir, target="content")
        self.assertEqual(len(hits), 1)
        self.assertIn("a.txt", hits[0])

    def test_search_filename(self):
        self.fs.write(self.p("report.md"), "x")
        hits = self.fs.search("report*", self.dir, target="filename")
        self.assertEqual(hits, ["report.md"])

    def test_search_no_matches(self):
        self.fs.write(self.p("a.txt"), "nothing")
        self.assertEqual(self.fs.search("zzz", self.dir), [])

    # ---- manage ----
    def test_make_directory_idempotent(self):
        self.fs.make_directory(self.p("d"))
        self.fs.make_directory(self.p("d"))  # exist_ok
        self.assertTrue(os.path.isdir(self.p("d")))

    def test_delete_file_and_dir(self):
        self.fs.write(self.p("a.txt"), "x")
        self.fs.delete(self.p("a.txt"))
        self.assertFalse(os.path.exists(self.p("a.txt")))
        self.fs.make_directory(self.p("d"))
        self.fs.delete(self.p("d"))
        self.assertFalse(os.path.exists(self.p("d")))

    def test_delete_missing_raises(self):
        with self.assertRaises(FileSystemError):
            self.fs.delete(self.p("nope"))

    def test_move(self):
        self.fs.write(self.p("a.txt"), "x")
        self.fs.move(self.p("a.txt"), self.p("b.txt"))
        self.assertFalse(os.path.exists(self.p("a.txt")))
        self.assertEqual(self.fs.read(self.p("b.txt")), "x")

    # ---- tree ----
    def test_tree_lists_files_dirs_and_sizes(self):
        self.fs.write(self.p("a.txt"), "hello")        # 5 bytes
        self.fs.make_directory(self.p("sub"))
        self.fs.write(self.p("sub", "nested.txt"), "x")
        out = self.fs.tree(self.dir)
        self.assertIn("a.txt", out)
        self.assertIn("sub/", out)                     # dir marker
        self.assertIn("(5B)", out)                     # file size
        self.assertIn("nested.txt", out)               # recursed into sub
        # dirs are listed before files at the same level
        self.assertLess(out.index("sub/"), out.index("a.txt"))

    def test_tree_empty_dir(self):
        self.assertEqual(self.fs.tree(self.dir), "(empty)")

    def test_tree_respects_max_entries(self):
        for i in range(20):
            self.fs.write(self.p(f"f{i:02d}.txt"), "x")
        out = self.fs.tree(self.dir, max_entries=5)
        self.assertEqual(out.count(".txt"), 5)
        self.assertIn("more", out)                     # truncation marker

    def test_tree_respects_max_depth(self):
        self.fs.write(self.p("deep", "x.txt"), "x")
        out = self.fs.tree(self.dir, max_depth=1)
        self.assertIn("deep/", out)
        self.assertNotIn("x.txt", out)                 # depth cap stops recursion
        self.assertIn("more", out)

    def test_tree_missing_dir_raises(self):
        with self.assertRaises(FileSystemError):
            self.fs.tree(self.p("nope"))


if __name__ == "__main__":
    unittest.main()
