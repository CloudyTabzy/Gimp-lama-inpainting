#!lama-gimp-python
"""GIMP 3.x plug-in for LaMa inpainting through a sidecar worker.

Exposes a single image-scoped procedure:

- ``plug-in-lama-inpaint`` — single-pass LaMa FFC model. ~2 s via
  the optional Rust ORT CPU sidecar; the Python ORT sidecar is the
  default when the Rust binary is not installed.

The worker lives in this same plug-in directory and uses the same
GIMP-side Python interpreter (GIMP's bundled MINGW Python 3.14, via
the per-user ``.interp`` mapping installed by ``install.bat``). ML
inference happens out of process — either in the Python ORT sidecar
or, when present, the Rust ORT sidecar.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import gi
gi.require_version('Gegl', '0.4')
gi.require_version('Gimp', '3.0')
gi.require_version('GimpUi', '3.0')

from gi.repository import Gegl, Gimp, GimpUi, GLib, GObject


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "lama.log"))
# Max number of lines to keep in the log file. Older lines are
# truncated on each write so the log doesn't grow without bound
# across many inpaint runs.
LOG_MAX_LINES = 200


# LaMa (single-pass FFC, ~2 s)
WORKER_SCRIPT = os.path.abspath(os.path.join(PLUGIN_DIR, "lama_worker.py"))
RUST_WORKER_BINARY = os.path.abspath(os.path.join(PLUGIN_DIR, "lama_worker_rust.exe"))
MODEL_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "lama_fp32.onnx"))
MANGA_MODEL_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "lama-manga.safetensors"))
CONFIG_PATH = os.path.abspath(os.path.join(PLUGIN_DIR, "lama_config.json"))

WORKER_TIMEOUT_SECONDS = 300
WORKER_POLL_INTERVAL_SECONDS = 0.25


def _log(msg):
    """Append a line to the plug-in log file.

    GIMP does not relay plug-in stdout/stderr to any visible console in
    GUI mode (``G_SPAWN_LEAVE_DESCRIPTORS_OPEN`` inherits GIMP's own
    descriptors, which are null for a normal desktop launch). Writing to
    a file instead guarantees the information is always available.

    To prevent stale results from accumulating across many inpaint
    runs, we truncate the log on each call to keep only the most
    recent ``LOG_MAX_LINES`` entries. This makes the log useful for
    debugging the last few runs without growing without bound.
    """
    try:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        # Read existing content (if any), append the new line, and
        # truncate to the most recent LOG_MAX_LINES entries.
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                existing = f.read().splitlines()
        except OSError:
            existing = []
        existing.append(line.rstrip("\n"))
        if len(existing) > LOG_MAX_LINES:
            existing = existing[-LOG_MAX_LINES:]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(existing) + "\n")
    except OSError:
        pass


def _normalize_python_path(path, relative_to=None):
    if not isinstance(path, str) or not path.strip():
        return None
    path = path.strip().strip('"')
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isabs(path) and relative_to:
        path = os.path.join(relative_to, path)
    return os.path.abspath(path)


def _configured_python():
    if not os.path.isfile(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError, TypeError):
        return None
    return _normalize_python_path(config.get("worker_python"), PLUGIN_DIR)


def find_worker_python():
    """Find the worker interpreter in deterministic preference order."""
    candidates = []

    configured = _configured_python()
    if configured:
        candidates.append(configured)

    environment_python = _normalize_python_path(
        os.environ.get("LAMA_WORKER_PYTHON")
    )
    if environment_python:
        candidates.append(environment_python)

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            for minor in range(14, 9, -1):
                candidates.append(
                    os.path.join(
                        local_app_data,
                        "Programs",
                        "Python",
                        f"Python3{minor}",
                        "python.exe",
                    )
                )

    for command in ("python", "python3"):
        discovered = shutil.which(command)
        if discovered:
            candidates.append(discovered)

    if sys.executable:
        candidates.append(sys.executable)

    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        key = os.path.normcase(candidate)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "No worker Python was found. Run install.bat with a Python 3.10+ "
        "path or set LAMA_WORKER_PYTHON to python.exe."
    )


def find_rust_worker():
    """Locate the opt-in Rust worker binary next to the plug-in.

    Returns the absolute path to the binary when it exists and is a
    regular file, otherwise ``None``. The caller is expected to fall
    back to the Python worker when this returns ``None``; the
    plug-in must remain fully functional without the Rust binary
    present.
    """
    if os.path.isfile(RUST_WORKER_BINARY):
        return RUST_WORKER_BINARY
    return None


def _configured_worker_kind():
    """Read the optional ``worker_kind`` field from ``lama_config.json``."""
    if not os.path.isfile(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("worker_kind")
    return value if isinstance(value, str) else None


def _env_truthy(name):
    """Return True iff the named env var is set to a truthy string."""
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    return None


def use_rust_worker():
    """Decide whether the plug-in should invoke the Rust worker.

    Resolution order (first match wins):

    1. ``LAMA_USE_RUST_WORKER`` env var. ``1``/``true``/``yes``/``on``
       opt in, ``0``/``false``/``no``/``off`` opt out, empty / unset
       falls through to step 2.
    2. ``worker_kind`` field in ``lama_config.json``. ``"rust"`` opts
       in, ``"python"`` opts out, anything else falls through to the
       default.
    3. Default: Python worker.

    The function never raises. A request for the Rust worker that
    does not have the binary present is silently treated as Python
    so the plug-in remains functional in a Rust-less install.
    """
    rust_binary = find_rust_worker()
    if rust_binary is None:
        return False

    truthy = _env_truthy("LAMA_USE_RUST_WORKER")
    if truthy is True:
        return True
    if truthy is False:
        return False

    config_value = _configured_worker_kind()
    if config_value is not None:
        normalized = config_value.strip().lower()
        if normalized == "rust":
            return True
        if normalized == "python":
            return False

    # Default: Rust worker when binary exists, Python when absent.
    # `find_rust_worker()` already confirmed the binary exists above
    # (line 172 returns False early if missing), so at this point
    # the binary is present and we prefer it. The Python worker is
    # the fallback when no Rust binary is installed.
    return True


def _run_subprocess(command):
    """Short-lived probe-style subprocess. Uses subprocess.run as documented."""
    kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": WORKER_TIMEOUT_SECONDS,
    }
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(command, **kwargs)


def _process_detail(completed):
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return f"exit code {completed.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = lines[-1] if lines else output
    return detail[:500]


def _windows_no_window_popen_kwargs():
    """Build kwargs that hide the spawned worker console on Windows.

    Both ``CREATE_NO_WINDOW`` and a ``STARTUPINFO`` with
    ``STARTF_USESHOWWINDOW``/``SW_HIDE`` are applied because some Python
    builds on Windows expose one flag but not the other. This only hides
    the *spawned ML worker* console; the terminal used to launch GIMP
    with ``--verbose`` remains visible.
    """
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "bufsize": 0,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name != "nt":
        return kwargs

    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    startupinfo = None
    start_flags = getattr(subprocess, "STARTF_USESHOWWINDOW", None)
    sw_hide = getattr(subprocess, "SW_HIDE", None)
    if start_flags is not None and sw_hide is not None:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = start_flags
        startupinfo.wShowWindow = sw_hide
    if startupinfo is not None:
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _run_worker_with_progress(command, progress_callback):
    """Run the ML worker with ``Popen`` so the GIMP UI can stay responsive.

    Pipe draining is delegated to a daemon reader thread that reads the
    merged stdout/stderr line-by-line into a ``queue.Queue``. The main
    thread never blocks on the pipe directly: the worker emits only a
    few short marker lines and then spends its time in ONNX inference,
    so a direct read could sit waiting for the next chunk that does
    not arrive until the worker exits — that would freeze progress
    updates, the timeout check, and the ``finally`` cleanup all at
    once. With the reader thread the main thread can poll
    ``process.poll()`` continuously, drive the GIMP progress callback
    every ~0.25 s, and still enforce the 300-second deadline.

    * stdout and stderr are merged into a single pipe (``stderr=STDOUT``).
    * The reader thread is a daemon so it cannot block process exit; it
      closes the pipe handle itself in ``finally`` and signals completion
      with a ``None`` sentinel.
    * On timeout/exit the child is terminated and then killed if it
      does not exit, so the pipe closes and the reader thread can
      drain. The reader is then joined with a brief timeout.
    * Returns a ``SimpleNamespace`` with ``returncode``, ``stdout``,
      ``stderr`` and ``timed_out`` attributes — a
      ``subprocess.CompletedProcess``-like shape that the existing
      error path already consumes.
    * Preserves UTF-8 decoding with ``errors="replace"`` and the
      Windows hide-console flags from
      :func:`_windows_no_window_popen_kwargs`.
    """
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    popen_kwargs = _windows_no_window_popen_kwargs()
    process = subprocess.Popen(command, **popen_kwargs)

    output_queue: "queue.Queue[object]" = queue.Queue()

    def _reader():
        """Drain the merged pipe line-by-line into ``output_queue``.

        ``readline`` blocks between lines, which is fine because this
        thread is the only consumer of the pipe and is daemonic. When
        the worker exits (or is terminated/killed) the pipe closes,
        ``readline`` returns ``""``, the iterator terminates, and the
        ``finally`` block closes the pipe handle and signals completion
        with a ``None`` sentinel.
        """
        if process.stdout is None:
            output_queue.put(None)
            return
        try:
            for line in iter(process.stdout.readline, ""):
                output_queue.put(line)
        except Exception:
            pass
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass
            output_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = False
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                break

            if progress_callback is not None:
                try:
                    progress_callback()
                except Exception:
                    pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            time.sleep(min(WORKER_POLL_INTERVAL_SECONDS, remaining))
    finally:
        # Stop the child first so the pipe closes and the reader thread
        # can drain and exit on its own, then join the reader briefly.
        if process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass

        reader_thread.join(timeout=2.0)

    output_chunks = []
    while True:
        try:
            item = output_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            break
        output_chunks.append(item)

    return SimpleNamespace(
        returncode=process.returncode,
        stdout="".join(output_chunks),
        stderr=None,
        timed_out=timed_out,
    )


def check_worker_dependencies(python_path):
    """Return None when usable, otherwise a concise diagnostic string."""
    probe = (
        "import importlib, sys; "
        "sys.exit('Python 3.10+ is required') "
        "if sys.version_info < (3, 10) else None; "
        "[importlib.import_module(name) "
        "for name in ('numpy', 'onnxruntime', 'PIL')]"
    )
    try:
        completed = _run_subprocess([python_path, "-c", probe])
    except subprocess.TimeoutExpired:
        return "dependency check timed out after 300 seconds"
    except OSError as exc:
        return f"could not start the interpreter: {exc}"
    if completed.returncode != 0:
        return _process_detail(completed)
    return None


def save_buffer_as_png(buffer, path):
    """Save a GEGL buffer through buffer-source -> png-save."""
    graph = Gegl.Node()
    source = graph.create_child("gegl:buffer-source")
    source.set_property("buffer", buffer)
    saver = graph.create_child("gegl:png-save")
    saver.set_property("path", path)
    source.link(saver)
    saver.process()
    if not os.path.isfile(path):
        raise OSError(f"GEGL did not create {path}")


def save_drawable_selection_mask(image, drawable, width, height, path):
    """Copy the image-space selection into a drawable-local Y u8 buffer."""
    selection = image.get_selection()
    if selection is None:
        raise RuntimeError("GIMP did not return the active selection")

    success, offset_x, offset_y = drawable.get_offsets()
    if not success:
        raise RuntimeError("Could not read the drawable offset")

    selection_buffer = selection.get_buffer()
    local_mask = Gegl.Buffer.new("Y u8", 0, 0, width, height)
    source_rect = Gegl.Rectangle.new(offset_x, offset_y, width, height)
    destination_rect = Gegl.Rectangle.new(0, 0, width, height)
    selection_buffer.copy(
        source_rect,
        Gegl.AbyssPolicy.NONE,
        local_mask,
        destination_rect,
    )
    save_buffer_as_png(local_mask, path)


def load_png_into_shadow(drawable, path, expected_width, expected_height):
    """Load a PNG through png-load -> write-buffer into the shadow buffer."""
    shadow_buffer = drawable.get_shadow_buffer()
    graph = Gegl.Node()
    loader = graph.create_child("gegl:png-load")
    loader.set_property("path", path)

    bounds = loader.get_bounding_box()
    if bounds.width != expected_width or bounds.height != expected_height:
        raise ValueError(
            "worker result dimensions differ: "
            f"{bounds.width}x{bounds.height} vs "
            f"{expected_width}x{expected_height}"
        )

    writer = graph.create_child("gegl:write-buffer")
    writer.set_property("buffer", shadow_buffer)
    loader.link(writer)
    writer.process()
    shadow_buffer.flush()


def _return_error(procedure, status, message):
    error = GLib.Error.new_literal(Gimp.PlugIn.error_quark(), message, 0)
    return procedure.new_return_values(status, error)


def _calling_error(procedure, message):
    return _return_error(procedure, Gimp.PDBStatusType.CALLING_ERROR, message)


def _execution_error(procedure, message):
    return _return_error(procedure, Gimp.PDBStatusType.EXECUTION_ERROR, message)


def _safe_progress(callable_, *args, **kwargs):
    """Invoke a Gimp.progress_* function without aborting on failure.

    GIMP's progress API is straightforward in normal use, but wrapping
    every call keeps the plug-in functional if GIMP throws or if the
    PDB is in a non-interactive transition.
    """
    try:
        return callable_(*args, **kwargs)
    except Exception:
        return None


class LamaInpaint(Gimp.PlugIn):
    def do_set_i18n(self, _name):
        return False

    def do_query_procedures(self):
        return ["plug-in-lama-inpaint"]

    def do_create_procedure(self, name):
        if name == "plug-in-lama-inpaint":
            Gegl.init(None)
            procedure = Gimp.ImageProcedure.new(
                self,
                name,
                Gimp.PDBProcType.PLUGIN,
                self.run,
                None,
            )
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
            procedure.set_menu_label("_LaMa Inpaint...")
            procedure.set_icon_name(GimpUi.ICON_GEGL)
            procedure.add_menu_path("<Image>/Filters/Enhance/")

            # Model selection choice.
            choice = Gimp.Choice.new()
            choice.add("lama", 0, "_LaMa (general)", "")
            if os.path.isfile(MANGA_MODEL_PATH):
                choice.add("manga", 1, "_Manga (line art)", "")
            procedure.add_choice_argument(
                "model",
                "Mo_del",
                "Inpainting model to use",
                choice,
                "lama",
                GObject.ParamFlags.READWRITE,
            )

            procedure.set_documentation(
                "Inpaint the active selection with the LaMa model "
                "(single-pass, ~2 s).",
                "Exports the drawable and selection to a sidecar worker, then "
                "applies the inpainted result to the active selection. "
                "Color preservation is exact outside the selection; "
                "best for removing spots, wires, and small blemishes.",
                name,
            )
            procedure.set_attribution(
                "GIMP Inpainting Plug-in",
                "GIMP Inpainting Plug-in",
                "2026",
            )
            return procedure
        return None

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        try:
            if run_mode == Gimp.RunMode.INTERACTIVE:
                dialog = GimpUi.ProcedureDialog.new(procedure, config)
                dialog.fill(["model"])
                if not dialog.run():
                    dialog.destroy()
                    return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
                dialog.destroy()

            model_choice = config.get_property("model")
            _log(f"run: model_choice={model_choice}, run_mode={run_mode}")
            return self._run_lama(procedure, run_mode, image, drawables, model_choice)
        except Exception as exc:
            _log(f"run EXCEPTION: {exc}")
            return _execution_error(procedure, f"LaMa Inpaint error: {exc}")

    # ----------------- LaMa backend -----------------

    def _run_lama(self, procedure, run_mode, image, drawables, model_choice="lama"):
        # Progress is started exactly once after the cheap pre-flight
        # checks have all passed, and is always ended in ``finally`` so
        # the GIMP progress bar cannot be left in a half-state on any
        # failure path (timeout, exception, cancelled run).
        progress_started = False

        def _phase(text, fraction):
            if not progress_started:
                return
            _safe_progress(Gimp.progress_set_text, text)
            _safe_progress(Gimp.progress_update, fraction)

        def _pulse():
            if not progress_started:
                return
            _safe_progress(Gimp.progress_pulse)
            _safe_progress(
                Gimp.progress_set_text,
                "Running LaMa inference (NN opaque)... please wait",
            )

        def _worker_progress_callback():
            _pulse()

        try:
            if len(drawables) != 1:
                return _calling_error(
                    procedure,
                    f"LaMa Inpaint requires exactly one drawable; got {len(drawables)}.",
                )

            drawable = drawables[0]
            intersects, selection_x, selection_y, selection_width, selection_height = (
                drawable.mask_intersect()
            )
            if not intersects:
                return _calling_error(
                    procedure,
                    "Make a non-empty selection on the active drawable first.",
                )

            width = drawable.get_width()
            height = drawable.get_height()
            if width <= 0 or height <= 0:
                return _calling_error(procedure, "The active drawable is empty.")

            # Select model based on user choice.
            if model_choice == "manga" and os.path.isfile(MANGA_MODEL_PATH):
                model_path = MANGA_MODEL_PATH
            else:
                model_path = MODEL_PATH

            if not os.path.isfile(model_path):
                return _calling_error(
                    procedure,
                    f"Model is missing: {model_path}. Reinstall the plug-in.",
                )

            # Worker selection. The Rust worker is an opt-in drop-in
            # for the Python worker. It is never required: when the
            # opt-in is set but the binary is missing we fall back to
            # the Python worker silently. The Python worker script
            # check, the interpreter probe, and the dependency probe
            # are skipped entirely on the Rust path.
            rust_binary = find_rust_worker()
            use_rust = use_rust_worker() and rust_binary is not None
            worker_python = None

            if not use_rust:
                if not os.path.isfile(WORKER_SCRIPT):
                    return _calling_error(
                        procedure,
                        f"Worker script is missing: {WORKER_SCRIPT}. "
                        "Reinstall the plug-in.",
                    )

                try:
                    worker_python = find_worker_python()
                except RuntimeError as exc:
                    return _calling_error(procedure, str(exc))

                dependency_error = check_worker_dependencies(worker_python)
                if dependency_error:
                    return _calling_error(
                        procedure,
                        "The worker Python is not ready:\n"
                        f"  {worker_python}\n"
                        f"  {dependency_error}\n\n"
                        "Install the dependencies with:\n"
                        f'  "{worker_python}" -m pip install pillow numpy onnxruntime\n'
                        "Then rerun install.bat or set LAMA_WORKER_PYTHON.",
                    )

            # All pre-flight checks passed; start the progress bar.
            _safe_progress(Gimp.progress_init, "LaMa Inpaint")
            progress_started = True
            _phase("Preparing image...", 0.05)

            try:
                with tempfile.TemporaryDirectory(prefix="gimp-lama-") as temp_dir:
                    image_path = os.path.join(temp_dir, "image.png")
                    mask_path = os.path.join(temp_dir, "mask.png")
                    output_path = os.path.join(temp_dir, "result.png")

                    _phase("Preparing image...", 0.10)
                    save_buffer_as_png(drawable.get_buffer(), image_path)

                    _phase("Exporting mask...", 0.20)
                    save_drawable_selection_mask(
                        image,
                        drawable,
                        width,
                        height,
                        mask_path,
                    )

                    if use_rust:
                        # The Rust binary takes the same CLI as the
                        # Python worker, with no Python interpreter in
                        # front of it.
                        command = [
                            rust_binary,
                            "--image",
                            image_path,
                            "--mask",
                            mask_path,
                            "--output",
                            output_path,
                            "--model",
                            model_path,
                        ]
                    else:
                        command = [
                            worker_python,
                            WORKER_SCRIPT,
                            "--image",
                            image_path,
                            "--mask",
                            mask_path,
                            "--output",
                            output_path,
                            "--model",
                            model_path,
                        ]

                    _phase("Starting LaMa worker...", 0.30)

                    if use_rust:
                        _log("worker: rust")
                    else:
                        _log("worker: python (%s)" % worker_python)

                    try:
                        completed = _run_worker_with_progress(
                            command, _worker_progress_callback
                        )
                    except OSError as exc:
                        return _execution_error(
                            procedure,
                            f"Could not start the LaMa worker: {exc}",
                        )

                    if completed.timed_out:
                        return _execution_error(
                            procedure,
                            "The LaMa worker timed out after "
                            f"{WORKER_TIMEOUT_SECONDS} seconds.",
                        )
                    if completed.returncode != 0:
                        return _execution_error(
                            procedure,
                            "The LaMa worker failed: "
                            + _process_detail(completed),
                        )
                    if not os.path.isfile(output_path):
                        return _execution_error(
                            procedure,
                            "The LaMa worker did not create its output PNG.",
                        )

                    # Parse the worker's provider marker when Rust was used.
                    if use_rust:
                        for line in (completed.stdout or "").splitlines():
                            if "provider:" in line:
                                provider = line.split("provider:")[-1].strip()
                                _log("provider: %s" % provider)
                                break

                    _phase("Applying result...", 0.90)
                    load_png_into_shadow(drawable, output_path, width, height)

                drawable.merge_shadow(True)
                drawable.update(
                    selection_x,
                    selection_y,
                    selection_width,
                    selection_height,
                )
                Gimp.displays_flush()
            except (OSError, RuntimeError, ValueError) as exc:
                return _execution_error(procedure, f"LaMa image transfer failed: {exc}")
            except Exception as exc:
                return _execution_error(procedure, f"LaMa Inpaint failed: {exc}")

            _phase("Complete", 1.0)
            return procedure.new_return_values(
                Gimp.PDBStatusType.SUCCESS,
                GLib.Error(),
            )
        finally:
            if progress_started:
                _safe_progress(Gimp.progress_end)


Gimp.main(LamaInpaint.__gtype__, sys.argv)
