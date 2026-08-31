import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

SPECIFICATION = importlib.util.spec_from_file_location(
    "arbuz_install", REPOSITORY_ROOT / "scripts" / "arbuz_install.py"
)
arbuz_install = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(arbuz_install)


class SettingsFileTests(unittest.TestCase):
    """webui.settings.bat is how the installer talks to webui.bat."""

    def setUp(self):
        self.text = arbuz_install.settings_text()

    def test_points_the_launcher_at_the_built_environment(self):
        self.assertIn('set "VENV_DIR=%~dp0installer_files\\env"', self.text)

    def test_every_path_is_relative_to_the_bat_file(self):
        # %~dp0 means the folder the bat lives in, so the whole thing keeps
        # working after the folder is moved to another drive.
        for line in self.text.splitlines():
            if line.startswith("set \""):
                path = line.split("=", 1)[1]
                self.assertTrue(path.startswith("%~dp0"), line)
                self.assertNotIn(":\\", path.replace("%~dp0", ""), line)

    def test_redirects_the_caches_that_usually_grow_on_the_system_drive(self):
        for name in ("PIP_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TEMP", "TMP"):
            with self.subTest(name=name):
                self.assertIn(name, self.text)

    def test_uses_windows_line_endings(self):
        self.assertIn("\r\n", self.text)
        self.assertFalse(self.text.replace("\r\n", "").startswith("\n"))

    def test_is_ascii_so_no_codepage_can_garble_it(self):
        self.text.encode("ascii")

    def test_opens_the_ui_in_the_browser(self):
        self.assertIn("set COMMANDLINE_ARGS=--autolaunch", self.text)

    def test_writing_keeps_the_old_file_under_bak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, original = root / "webui.settings.bat", arbuz_install.SETTINGS_FILE
            settings.write_text("set COMMANDLINE_ARGS=--medvram", encoding="utf8")
            try:
                arbuz_install.SETTINGS_FILE = settings
                backup = arbuz_install.write_settings()
            finally:
                arbuz_install.SETTINGS_FILE = original
            self.assertIsNotNone(backup)
            self.assertEqual(backup.read_text(encoding="utf8"), "set COMMANDLINE_ARGS=--medvram")
            self.assertIn("VENV_DIR", settings.read_text(encoding="utf8"))

    def test_writing_reports_no_backup_when_there_was_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, original = Path(directory) / "webui.settings.bat", arbuz_install.SETTINGS_FILE
            try:
                arbuz_install.SETTINGS_FILE = settings
                self.assertIsNone(arbuz_install.write_settings())
            finally:
                arbuz_install.SETTINGS_FILE = original
            self.assertTrue(settings.exists())


class CacheEnvTests(unittest.TestCase):
    def test_every_cache_lives_inside_the_installation_folder(self):
        root = Path(tempfile.gettempdir()) / "arbuz-fake-root"
        for name, value in arbuz_install.cache_env(root).items():
            with self.subTest(name=name):
                self.assertEqual(Path(value).resolve().parts[: len(root.resolve().parts)],
                                 root.resolve().parts)
                self.assertNotEqual(Path(value).resolve(), root.resolve())

    def test_covers_pip_huggingface_and_temp(self):
        env = arbuz_install.cache_env()
        self.assertIn("pip", env["PIP_CACHE_DIR"])
        self.assertIn("huggingface", env["HF_HOME"])
        self.assertIn("tmp", env["TEMP"])
        self.assertEqual(env["TEMP"], env["TMP"])


class PinnedVersionsTests(unittest.TestCase):
    def test_reads_the_pins_out_of_launch_utils(self):
        pins = arbuz_install.pinned_versions()
        self.assertTrue(pins["torch"].count(".") >= 2)
        self.assertTrue(pins["cuda"].isdigit())
        self.assertTrue(pins["torchvision"].count(".") >= 2)
        self.assertTrue(pins["index"].startswith("http"))

    def test_falls_back_when_the_source_is_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "modules").mkdir()
            (root / "modules" / "launch_utils.py").write_text("def prepare_environment():\n    pass\n")
            pins = arbuz_install.pinned_versions(root)
        self.assertEqual(pins["torch"], arbuz_install.FALLBACK_TORCH)
        self.assertEqual(pins["index"], arbuz_install.FALLBACK_TORCH_INDEX)

    def test_install_command_carries_the_pins_and_the_index(self):
        command = arbuz_install.torch_install_command()
        pins = arbuz_install.pinned_versions()
        self.assertIn(f'torch=={pins["torch"]}+cu{pins["cuda"]}', command)
        self.assertIn(f'torchvision=={pins["torchvision"]}+cu{pins["cuda"]}', command)
        self.assertIn(pins["index"], command)

    def test_install_command_accepts_another_index(self):
        command = arbuz_install.torch_install_command(index="https://example.invalid/whl")
        self.assertIn("https://example.invalid/whl", command)
        self.assertNotIn(arbuz_install.FALLBACK_TORCH_INDEX, command)


class LanguageTests(unittest.TestCase):
    def test_explicit_choice_is_kept(self):
        self.assertEqual(arbuz_install.pick_language("ru"), "ru")
        self.assertEqual(arbuz_install.pick_language("en"), "en")

    def test_auto_follows_the_console_encoding(self):
        class FakeStdout:
            def __init__(self, encoding):
                self.encoding = encoding

        for encoding, expected in (("utf-8", "ru"), ("cp866", "en"), ("cp1251", "en"), ("", "en")):
            with self.subTest(encoding=encoding):
                with mock.patch.object(sys, "stdout", FakeStdout(encoding)):
                    self.assertEqual(arbuz_install.pick_language("auto"), expected)

    def test_every_message_exists_in_both_languages(self):
        for key, (russian, english) in arbuz_install.TEXT.items():
            with self.subTest(key=key):
                self.assertTrue(russian and english)
                self.assertNotEqual(russian, english)

    def test_t_picks_the_requested_language(self):
        self.assertEqual(arbuz_install.t("done", "ru"), arbuz_install.TEXT["done"][0])
        self.assertEqual(arbuz_install.t("done", "en"), arbuz_install.TEXT["done"][1])


class HelperTests(unittest.TestCase):
    def test_quote_leaves_launcher_commands_alone(self):
        self.assertEqual(arbuz_install.quote("py -3.11"), "py -3.11")

    def test_quote_wraps_real_paths(self):
        self.assertEqual(arbuz_install.quote(__file__), f'"{__file__}"')

    def test_venv_python_follows_the_platform(self):
        env = Path("env")  # built outside the patch: Path() itself dispatches on os.name
        with mock.patch.object(os, "name", "nt"):
            windows = arbuz_install.venv_python(env)
        with mock.patch.object(os, "name", "posix"):
            posix = arbuz_install.venv_python(env)
        self.assertEqual(windows, Path("env/Scripts/python.exe"))
        self.assertEqual(posix, Path("env/bin/python"))

    def test_reachable_is_false_when_the_host_refuses(self):
        class Exploding:
            def __enter__(self):
                raise OSError("no route to host")

            def __exit__(self, *args):
                return False

        with mock.patch.object(arbuz_install.urllib.request, "urlopen", lambda *a, **k: Exploding()):
            self.assertFalse(arbuz_install.reachable("https://example.invalid"))

    def test_reachable_is_true_when_the_host_answers(self):
        class Answering:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with mock.patch.object(arbuz_install.urllib.request, "urlopen", lambda *a, **k: Answering()):
            self.assertTrue(arbuz_install.reachable("https://example.invalid"))

    def test_download_reports_failure_without_leaving_a_broken_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "python.tar.gz"
            with mock.patch.object(arbuz_install.urllib.request, "urlopen", side_effect=OSError("blocked")):
                self.assertFalse(arbuz_install.download("https://example.invalid", target))
            self.assertFalse(target.exists())

    def test_free_bytes_is_reported(self):
        self.assertGreater(arbuz_install.free_bytes(), 0)


class LauncherFileTests(unittest.TestCase):
    """The .bat files are the whole point of the one click, so guard them."""

    def test_the_entry_points_exist(self):
        for name in ("install.bat", "run.bat", "console.bat"):
            with self.subTest(name=name):
                self.assertTrue((REPOSITORY_ROOT / name).is_file())

    def test_batch_files_are_crlf_and_ascii_only(self):
        for name in ("install.bat", "run.bat", "console.bat"):
            with self.subTest(name=name):
                data = (REPOSITORY_ROOT / name).read_bytes()
                self.assertIn(b"\r\n", data)
                self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
                data.decode("ascii")

    def test_install_bat_runs_the_installer_script(self):
        data = (REPOSITORY_ROOT / "install.bat").read_text(encoding="ascii")
        self.assertIn("scripts\\arbuz_install.py", data)

    def test_install_bat_resolves_launchers_to_a_path(self):
        # Quoting "py -3.11" makes cmd look for a program with that literal
        # name, so the candidate has to become a full path before it is quoted.
        data = (REPOSITORY_ROOT / "install.bat").read_text(encoding="ascii")
        self.assertIn("print(sys.executable)", data)
        self.assertIn('"%BOOTSTRAP%" scripts\\arbuz_install.py', data)
        self.assertNotIn('call :try_python', data)

    def test_install_bat_fetches_the_repository_when_it_is_alone(self):
        # People download the single file. It has to be able to finish the job.
        data = (REPOSITORY_ROOT / "install.bat").read_text(encoding="ascii")
        self.assertIn("if exist \"%~dp0scripts\\arbuz_install.py\" goto :have_repo", data)
        self.assertIn("archive/refs/heads/", data)
        self.assertIn("--strip-components=1", data)
        self.assertIn("repo_failed", data)

    def test_install_bat_does_not_require_a_python_installation(self):
        data = (REPOSITORY_ROOT / "install.bat").read_text(encoding="ascii")
        self.assertIn("curl.exe", data)
        self.assertIn("tar.exe", data)
        self.assertIn("installer_files\\python", data)

    def test_run_bat_checks_the_environment_before_launching(self):
        data = (REPOSITORY_ROOT / "run.bat").read_text(encoding="ascii")
        self.assertIn("installer_files\\env\\Scripts\\python.exe", data)
        self.assertIn("call webui.bat", data)

    def test_both_files_carry_the_same_revision(self):
        # pasted logs are only useful if they say which build produced them
        import re
        source = (REPOSITORY_ROOT / "scripts" / "arbuz_install.py").read_text(encoding="utf8")
        revision = re.search(r'^INSTALLER_REV = "(\d+)"', source, re.M).group(1)
        self.assertIn(f'set "REV={revision}"', (REPOSITORY_ROOT / "install.bat").read_text(encoding="ascii"))
        self.assertIn("[rev %REV%]", (REPOSITORY_ROOT / "install.bat").read_text(encoding="ascii"))
        self.assertIn("[rev {INSTALLER_REV}]", source)

    def test_doctor_mode_exists(self):
        # the only way to debug a machine you cannot see
        source = (REPOSITORY_ROOT / "scripts" / "arbuz_install.py").read_text(encoding="utf8")
        self.assertIn('"--doctor"', source)
        self.assertIn("def doctor(", source)

    def test_the_installer_folder_is_ignored_by_git(self):
        ignored = (REPOSITORY_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("/installer_files/", ignored)
        self.assertIn("/webui.settings.bat", ignored)


if __name__ == "__main__":
    unittest.main()
