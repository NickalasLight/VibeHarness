import os
import tempfile
import unittest

from vibeharness.filesystem import PAGE_SIZE, FileSystem, FileSystemError


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

    # ---- paged read ----
    def test_read_page_small_is_single_page(self):
        self.fs.write(self.p("a.txt"), "hello")
        page = self.fs.read_page(self.p("a.txt"))
        self.assertEqual(page.text, "hello")
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.total_pages, 1)
        self.assertEqual(page.total_chars, 5)

    def test_read_page_multi_page(self):
        self.fs.write(self.p("a.txt"), "x" * (PAGE_SIZE * 2 + 5))
        p1 = self.fs.read_page(self.p("a.txt"), 1)
        self.assertEqual(len(p1.text), PAGE_SIZE)
        self.assertEqual(p1.total_pages, 3)
        self.assertEqual(p1.total_chars, PAGE_SIZE * 2 + 5)
        p3 = self.fs.read_page(self.p("a.txt"), 3)
        self.assertEqual(len(p3.text), 5)
        self.assertEqual(p3.page_number, 3)

    def test_read_page_out_of_range_raises(self):
        self.fs.write(self.p("a.txt"), "small")
        with self.assertRaises(FileSystemError):
            self.fs.read_page(self.p("a.txt"), 2)
        with self.assertRaises(FileSystemError):
            self.fs.read_page(self.p("a.txt"), 0)

    def test_read_page_empty_file_is_one_page(self):
        self.fs.write(self.p("a.txt"), "")
        page = self.fs.read_page(self.p("a.txt"))
        self.assertEqual(page.total_pages, 1)
        self.assertEqual(page.text, "")

    # ---- copy ----
    def test_copy_file_preserves_bytes_over_cap(self):
        big = "z" * (PAGE_SIZE * 3)  # larger than any observation cap
        self.fs.write(self.p("src.txt"), big)
        self.fs.copy(self.p("src.txt"), self.p("dst.txt"))
        self.assertEqual(self.fs.read(self.p("src.txt")), big)   # original preserved
        self.assertEqual(self.fs.read(self.p("dst.txt")), big)   # full bytes copied

    def test_copy_directory(self):
        self.fs.write(self.p("d", "a.txt"), "one")
        self.fs.write(self.p("d", "sub", "b.txt"), "two")
        self.fs.copy(self.p("d"), self.p("d2"))
        self.assertEqual(self.fs.read(self.p("d2", "a.txt")), "one")
        self.assertEqual(self.fs.read(self.p("d2", "sub", "b.txt")), "two")
        self.assertTrue(os.path.isdir(self.p("d")))  # original preserved

    def test_copy_auto_creates_dest_parent(self):
        self.fs.write(self.p("src.txt"), "x")
        self.fs.copy(self.p("src.txt"), self.p("new", "deep", "dst.txt"))
        self.assertEqual(self.fs.read(self.p("new", "deep", "dst.txt")), "x")

    def test_copy_missing_source_raises(self):
        with self.assertRaises(FileSystemError):
            self.fs.copy(self.p("nope.txt"), self.p("dst.txt"))

    def test_copy_dest_exists_raises(self):
        self.fs.write(self.p("src.txt"), "x")
        self.fs.write(self.p("dst.txt"), "y")
        with self.assertRaises(FileSystemError):
            self.fs.copy(self.p("src.txt"), self.p("dst.txt"))

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


if __name__ == "__main__":
    unittest.main()
