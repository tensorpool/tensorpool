import os
import time
from typing import Final, Optional, List, Dict, Tuple
import requests
from tqdm import tqdm
import importlib.metadata
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess
import sys
import asyncio
import websockets
import threading
from .spinner import Spinner
import platform

ENGINE: Final = os.environ.get("TENSORPOOL_ENGINE", "https://engine.tensorpool.dev")


def _run_streaming_command(
    command: str, show_stdout: bool = False
) -> Tuple[int, str, str]:
    """Run a shell command while preserving carriage-return progress updates."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        bufsize=0,
        env=env,
    )

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _drain_stream(stream, sink, target_stream=None):
        while True:
            chunk = stream.read(1)
            if chunk == b"":
                break
            decoded_chunk = chunk.decode("utf-8", errors="replace")
            sink.append(decoded_chunk)
            if target_stream is not None:
                if hasattr(target_stream, "buffer"):
                    target_stream.buffer.write(chunk)
                else:
                    target_stream.write(decoded_chunk)
                target_stream.flush()

    stdout_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stdout, stdout_chunks, sys.stdout if show_stdout else None),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stderr, stderr_chunks, sys.stderr if show_stdout else None),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    return returncode, "".join(stdout_chunks), "".join(stderr_chunks)


def safe_input(
    prompt: str, default: Optional[str] = None, no_input: bool = False
) -> str:
    """
    Safe input function that respects the --no-input flag.

    Args:
        prompt: The prompt to show to the user
        default: The default value to use when no_input is True
        no_input: Whether to skip interactive input

    Returns:
        The user input or default value

    Raises:
        SystemExit: When no_input is True and no default is provided
    """
    if no_input:
        if default is not None:
            print(f"{prompt.rstrip()}: {default}")
            return default
        else:
            print(f"Error: {prompt.rstrip(': ')} required but --no-input flag was used")
            exit(1)

    # Check if stdin is a TTY (interactive terminal)
    if not sys.stdin.isatty():
        if default is not None:
            print(f"{prompt.rstrip()}: {default} (non-interactive, using default)")
            return default
        else:
            print(
                f"Error: {prompt.rstrip(': ')} required but running in non-interactive mode"
            )
            exit(1)

    try:
        return input(prompt)
    except EOFError:
        if default is not None:
            print(f"\n{prompt.rstrip()}: {default} (EOF, using default)")
            return default
        else:
            print(f"\nError: {prompt.rstrip(': ')} required but stdin closed")
            exit(1)


def safe_confirm(prompt: str, no_input: bool = False, default: str = "y") -> str:
    """
    Safe confirmation function that respects the --no-input flag.

    Args:
        prompt: The confirmation prompt to show to the user
        no_input: Whether to skip interactive input
        default: The default response when no_input is True

    Returns:
        The user input or default confirmation
    """
    if no_input:
        print(f"{prompt.rstrip()}: {default}")
        return default

    # Check if stdin is a TTY (interactive terminal)
    if not sys.stdin.isatty():
        print(f"{prompt.rstrip()}: {default} (non-interactive, using default)")
        return default

    try:
        return input(prompt)
    except EOFError:
        print(f"\n{prompt.rstrip()}: {default} (EOF, using default)")
        return default


def _get_headers(
    include_auth: bool = True, content_type: str = "application/json"
) -> Dict[str, str]:
    """
    Get standard headers for API requests with X-Client-Type automatically included.
    Args:
        include_auth: Whether to include Authorization header (default True)
        content_type: Content-Type header value (default "application/json")
    Returns:
        Dictionary of headers with X-Client-Type: "cli" always included
    """
    headers = {
        "X-Client-Type": "cli",
        "Content-Type": content_type,
    }

    if include_auth:
        api_key = get_tensorpool_key()
        assert api_key is not None, "TENSORPOOL_KEY not found. Please set your API key."
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def get_tensorpool_key():
    """Get API key from env var first, then .env in cwd"""
    key = os.environ.get("TENSORPOOL_KEY")
    if key:
        return key

    try:
        with open(os.path.join(os.getcwd(), ".env")) as f:
            for line in f:
                if line.startswith("TENSORPOOL_KEY"):
                    return line.split("=", 1)[1].strip().strip("'").strip('"')
    except FileNotFoundError:
        return None

    return None


def save_tensorpool_key(api_key: str) -> bool:
    """Save API key to .env in current directory and set in environment"""
    try:
        with open(os.path.join(os.getcwd(), ".env"), "a+") as f:
            f.write(f"\nTENSORPOOL_KEY={api_key}\n")
        os.environ["TENSORPOOL_KEY"] = api_key
        assert os.getenv("TENSORPOOL_KEY") == api_key
        return True
    except Exception as e:
        print(f"Failed to save API key: {e}")
        return False


def login(no_input: bool = False):
    """
    Store the API key in the .env file and set it in the environment variables.
    """
    print("https://tensorpool.dev/dashboard")
    api_key = safe_input("Enter your TensorPool API key: ", no_input=no_input).strip()

    if not api_key:
        print("API key cannot be empty")
        return

    return save_tensorpool_key(api_key)


def get_version():
    try:
        return importlib.metadata.version(__package__)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def health_check() -> (bool, str):
    """
    Checks if the TensorPool engine is online and if the package version is acceptable.
    Returns:
        bool: If the user can proceed
        str: A message to display to the user
    """

    try:
        version = get_version()
        # print(f"Package version: {version}")
        headers = _get_headers()
        response = requests.post(
            f"{ENGINE}/health",
            json={
                "package_version": version,
                "uname": platform.uname(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_compiler": platform.python_compiler(),
                "python_build": platform.python_build(),
            },
            headers=headers,
            timeout=15,
        )
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            # Malformed response handling
            # print(response.text)
            return (
                False,
                f"Received malformed response from server during health check. Status code: {response.status_code} \nIf this persists, please contact team@tensorpool.dev",
            )

        msg = data.get("message")

        if response.status_code == 200:
            # Healthy
            return (True, msg)
        else:
            # Engine online, but auth or health check failure
            return (False, msg)
    except requests.exceptions.ConnectionError:
        return (
            False,
            "Cannot reach the TensorPool. Please check your internet connection.\nHaving trouble? Contact team@tensorpool.dev",
        )
    except Exception as e:
        # Catch-all for unexpected failures
        return (False, f"Unexpected error during health check: {str(e)}")


def _normalize_platform_system() -> str:
    """Return the lowercase OS slug expected by REST endpoints."""
    return {
        "Windows": "windows",
        "Linux": "linux",
        "Darwin": "darwin",
    }.get(platform.system(), platform.system().lower())


def _decode_response_json(response: requests.Response) -> Tuple[Optional[Dict], Optional[str]]:
    """Decode a JSON response into a dict and surface malformed responses consistently."""
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        return None, f"Failed to decode server response. Status code: {response.status_code}"

    if not isinstance(data, dict):
        return None, f"Unexpected server response type. Status code: {response.status_code}"

    return data, None


def _response_message(
    data: Optional[Dict], default: str, *, fallback_error_key: bool = True
) -> str:
    """Pick the most useful user-facing message from a JSON response."""
    if not data:
        return default

    if data.get("message"):
        return str(data["message"])

    if fallback_error_key and data.get("error"):
        return str(data["error"])

    if data.get("external_message"):
        return str(data["external_message"])

    return default


def _confirm_destructive_action(action: str, target: str, no_input: bool) -> Tuple[bool, str]:
    """Confirm a destructive action locally before calling the API."""
    if no_input:
        return True, ""

    answer = safe_confirm(f"{action} {target}? [y/N]: ", no_input=no_input, default="n")
    if answer.strip().lower() not in {"y", "yes"}:
        return False, f"{action} cancelled"

    return True, ""


def _poll_request_until_terminal(
    request_id: str,
    headers: Dict[str, str],
    spinner: Optional[Spinner],
    pending_text: str,
) -> Tuple[bool, str]:
    """Poll a request resource until it reaches a terminal state."""
    while True:
        try:
            response = requests.get(
                f"{ENGINE}/request/info/{request_id}",
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to poll request status: {str(e)}"

        data, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 200:
            return False, _response_message(
                data,
                f"Error fetching request status. Status code: {response.status_code}",
            )

        request_status = str(data.get("status", "")).upper()
        message = _response_message(data, pending_text)

        if spinner:
            spinner.update_text(message)

        if request_status == "COMPLETED":
            return True, message
        if request_status == "FAILED":
            return False, message

        retry_after = response.headers.get("Retry-After")
        try:
            delay = max(1, int(retry_after)) if retry_after else 2
        except ValueError:
            delay = 2
        time.sleep(delay)


def _poll_multiple_requests_until_terminal(
    request_ids: List[str],
    headers: Dict[str, str],
    spinner: Optional[Spinner],
    pending_text: str,
) -> Tuple[bool, str]:
    """Poll multiple request IDs sequentially until all complete or any fail."""
    last_message = pending_text
    for request_id in request_ids:
        success, message = _poll_request_until_terminal(
            request_id=request_id,
            headers=headers,
            spinner=spinner,
            pending_text=pending_text,
        )
        last_message = message
        if not success:
            return False, message

    return True, last_message


def _poll_job_cancel_until_terminal(
    job_id: str, headers: Dict[str, str], spinner: Optional[Spinner]
) -> Tuple[bool, str]:
    """Poll job info until cancellation reaches a terminal state."""
    while True:
        try:
            response = requests.get(
                f"{ENGINE}/job/info/{job_id}",
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to poll job status: {str(e)}"

        data, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 200:
            return False, _response_message(
                data,
                f"Error fetching job info. Status code: {response.status_code}",
            )

        job_info = data.get("job_info") or {}
        status = str(job_info.get("status", "")).lower()
        message = _response_message(data, f"Job status: {status or 'unknown'}")

        if spinner:
            spinner.update_text(message)

        if status == "canceled":
            return True, message
        if status in {"completed", "error", "failed"}:
            return False, message

        time.sleep(2)


def dump_file(content: str, path: str) -> bool:
    """
    Save raw text content to the specified file path
    Args:
        content: The raw text content to save
        path: The path to save the file
    Returns:
        A boolean indicating success
    """
    try:
        with open(path, "w") as f:
            f.write(content)
        return os.path.exists(path)
    except Exception:
        return False


def get_proj_paths():
    """
    Returns a list of all file paths in the project directory.
    """
    # TODO: make this use shouldignore
    files = [
        os.path.normpath(os.path.join(dirpath, f))
        for (dirpath, dirnames, filenames) in os.walk(".")
        if not any(i in dirpath for i in IGNORE_FILE_SUFFIXES)
        for f in filenames
        if not any(f.endswith(i) for i in IGNORE_FILE_SUFFIXES)
    ]

    return files


def get_empty_tp_config() -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Fetch the default empty tp config from the job/init endpoint
    Returns:
        A tuple containing success status, empty config dict, and optional message
    """
    headers = _get_headers()

    try:
        response = requests.get(
            f"{ENGINE}/job/init",
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, None, f"Failed to fetch empty config: {str(e)}"

    try:
        res = response.json()
    except requests.exceptions.JSONDecodeError:
        return False, None, "Received malformed response from server"

    if response.status_code != 200:
        message = res.get("message", "Failed to fetch empty config")
        return False, None, message

    empty_tp_config = res.get("empty_tp_config")
    message = res.get("message")

    if not empty_tp_config:
        return False, None, "No empty config received from server"

    return True, empty_tp_config, message


def job_listen(job_id: str) -> Tuple[bool, str]:
    """
    Listen to a job's output stream via WebSocket
    Args:
        job_id: The ID of the job to listen to
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not job_id:
        return False, "Job ID is required"

    # Build endpoint
    endpoint = f"/job/listen/{job_id}"

    # Empty payload
    # payload = {}

    # Run the async function without a spinner so messages print directly
    success, message = asyncio.run(
        _ws_operation_async(
            endpoint=endpoint,
            spinner=None,
            payload=None,
            handle_user_input=False,
            success_message="Job listening completed",
            error_message="Job listening failed",
            unexpected_end_message=(
                "Connection ended unexpectedly while listening to job.\n"
                f"Check 'tp job info {job_id}' to see the current status."
            ),
        )
    )

    return success, message


async def _job_push_async(
    tp_config: str,
    api_key: str,
    cluster_id: str,
    teardown_cluster: bool = False,
) -> Tuple[bool, Optional[str]]:
    ws_url = (
        f"{ENGINE.replace('http://', 'ws://').replace('https://', 'wss://')}/job/push"
    )
    # print("ws_url:", ws_url)

    job_id = None

    try:
        async with websockets.connect(
            ws_url, ping_interval=5, ping_timeout=10
        ) as websocket:
            # First message: Send API key
            await websocket.send(json.dumps({"TENSORPOOL_KEY": api_key}))

            # Second message: Send job configuration
            initial_data = {
                "tp_config": tp_config,
                "system": platform.system(),
                "cluster_id": cluster_id,
                "teardown_cluster": teardown_cluster,
            }

            await websocket.send(json.dumps(initial_data))

            # Process messages from server
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                # print("recieved data:", data)

                # Capture job_id if present
                if "job_id" in data and not job_id:
                    job_id = data["job_id"]

                # Print status messages
                if "message" in data:
                    print(data["message"])

                # Execute commands sent by server
                if "command" in data:
                    command = data["command"]
                    if not command:
                        # Skip execution if command is None or empty
                        continue

                    show_stdout = data.get("command_show_stdout", False)
                    try:
                        returncode, stdout, stderr = _run_streaming_command(
                            command, show_stdout=show_stdout
                        )

                        # Print errors if command failed (and not already shown)
                        if returncode != 0 and not show_stdout:
                            if stderr:
                                print(f"Command error: {stderr}")
                            if stdout:
                                print(f"Command output: {stdout}")

                    except Exception as e:
                        print(f"Failed to execute command: {str(e)}")
                        stdout = ""
                        returncode = 1

                    # print(stdout)
                    # print(stderr)

                    # Send result back to server
                    response = {
                        "type": "command_result",
                        "command": command,
                        "exit_code": returncode,
                        "command_stdout": stdout,
                        "command_stderr": stderr,
                    }
                    await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed as e:
        # Code 1000 is normal/successful closure
        if e.code == 1000:
            return True, job_id

        # Code 1006 is abnormal closure - connection lost but operation may still be happening
        if e.code == 1006:
            print("Connection lost during job submission.")
            print("Check job status with `tp job list`. Job likely has to be resubmitted.")
            return False, job_id

        # Other codes indicate errors
        print(f"Job connection closed, code = {e.code}")
        if e.reason:
            print(e.reason)
        return False, job_id

    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {str(e)}")
        return False, None

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False, None

    # If we get here, the WebSocket closed normally
    return True, job_id


def job_push(
    tp_config_path: str,
    cluster_id: str,
    teardown_cluster: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Push a job
    Args:
        tp_config_path: Path to the tp config file
        cluster_id: Cluster ID to run the job on
        teardown_cluster: Whether to destroy the cluster after the job finishes
    Returns:
        Tuple[bool, Optional[str]]: (success status, job_id if available)
    """
    if not os.path.exists(tp_config_path):
        print(f"Config file not found: {tp_config_path}")
        return False, None

    try:
        with open(tp_config_path, "r") as f:
            tp_config = f.read()
    except Exception as e:
        print(f"Failed to read {tp_config_path}: {str(e)}")
        return False, None

    api_key = get_tensorpool_key()
    if not api_key:
        print("TENSORPOOL_KEY not found. Please set your API key.")
        return False, None

    # Run the async function
    return asyncio.run(
        _job_push_async(
            tp_config,
            api_key,
            cluster_id=cluster_id,
            teardown_cluster=teardown_cluster,
        )
    )


def job_pull(
    job_id: str,
    files: Optional[List[str]] = None,
    dry_run: bool = False,
    tensorpool_priv_key_path: Optional[str] = None,
    spinner: Optional[Spinner] = None,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Pull job output files or execute commands
    Args:
        job_id: The ID of the job to pull
        files: Optional list of specific files to pull
        dry_run: If True, only preview files without downloading
        tensorpool_priv_key_path: Path to tensorpool private key
        spinner: Optional spinner to pause while streaming live command output
    Returns:
        A tuple containing:
        - None and a message on failure
        - An empty download map and an optional message when there are no files to pull
        - A populated download map and an optional message on success
    """
    if not job_id:
        return None, "Job ID is required"

    headers = _get_headers()
    params = {
        "dry_run": dry_run,
        "system": _normalize_platform_system(),
    }
    if files:
        params["files"] = files

    # Add private key path if provided
    if tensorpool_priv_key_path:
        params["private_key_path"] = tensorpool_priv_key_path

    try:
        response = requests.get(
            f"{ENGINE}/job/pull/{job_id}", params=params, headers=headers, timeout=60
        )
    except requests.exceptions.RequestException as e:
        return None, f"Failed to pull job: {str(e)}"

    try:
        result = response.json()
        # print(result)
    except requests.exceptions.JSONDecodeError:
        return (
            None,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error pulling job. Status code: {response.status_code}"
        )
        return None, error_msg

    # Handle command execution if present (similar to job_listen)
    if "command" in result:
        command = result["command"]
        if not command:
            # Skip execution if command is None or empty
            pass
        else:
            show_stdout = result.get("command_show_stdout", False)

            try:
                if spinner:
                    spinner.pause()
                returncode, stdout_text, stderr_text = _run_streaming_command(
                    command, show_stdout=show_stdout
                )

                # Print errors if command failed and not already shown
                if returncode != 0 and not show_stdout:
                    if stderr_text:
                        print(f"Command error: {stderr_text}")
                    if stdout_text:
                        print(f"Command output: {stdout_text}")

            except Exception as e:
                print(f"Failed to execute command: {str(e)}")
                return None, f"Failed to execute command: {str(e)}"
            finally:
                if spinner:
                    spinner.resume()

    download_map = result.get("download_map")
    message = result.get("message")

    return download_map, message


def download_files(download_map: Dict[str, str], overwrite: bool = False) -> bool:
    """
    Given a download map of file paths to signed GET URLs, download each file in parallel.
    If the same files exists locally, append a suffix to the filename.
    """

    max_workers = min(os.cpu_count() * 2 if os.cpu_count() else 6, 6)
    successes = []
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        def _download_file(file_info):
            file_path, url = file_info
            headers = {"Content-Type": "application/octet-stream"}
            max_retries = 3
            base_delay = 1

            for retries in range(max_retries + 1):
                try:
                    response = requests.get(url, headers=headers, stream=True)
                    if response.status_code != 200:
                        if retries < max_retries:
                            delay = base_delay * (2**retries)
                            time.sleep(delay)
                            continue
                        return False, (file_path, response.status_code, response.text)

                    total_size = int(response.headers.get("content-length", 0))

                    if os.path.exists(file_path):
                        if overwrite:
                            print(f"Overwriting {file_path}")
                        else:
                            print(f"Skipping {file_path} - file already exists")
                            return True, (file_path, 200, "Skipped - file exists")

                    # Create directories for path if they don't exist
                    dir_name = os.path.dirname(file_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)

                    with open(file_path, "wb") as f:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc=f"Downloading {os.path.basename(file_path)}{' (attempt ' + str(retries + 1) + ')' if retries > 0 else ''}",
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                    return True, (file_path, response.status_code, "Success")

                except Exception as e:
                    if retries < max_retries:
                        delay = base_delay * (2**retries)
                        time.sleep(delay)
                        continue
                    return False, (file_path, "Exception", str(e))

        future_to_file = {
            executor.submit(_download_file, (file_path, url)): file_path
            for file_path, url in download_map.items()
        }

        for future in concurrent.futures.as_completed(future_to_file):
            success, result = future.result()
            if success:
                successes.append(result[0])
            else:
                failures.append(result)

    if failures:
        print("The following downloads failed:")
        for path, code, text in failures:
            print(f"{path}: Status {code} - {text}")
        return False

    return True


def job_cancel(job_id: str, no_input: bool = False, wait: bool = False) -> Tuple[bool, str]:
    """
    Cancel a job
    Args:
        job_id: The ID of the job to cancel
        no_input: Whether to skip interactive confirmation prompts
        wait: Whether to wait for the job to fully terminate before returning
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    assert job_id is not None, "A job ID is needed to cancel"
    if not no_input and not sys.stdin.isatty():
        return False, "Cancel job requires explicit confirmation in non-interactive mode."

    confirmed, message = _confirm_destructive_action("Cancel job", job_id, no_input)
    if not confirmed:
        return False, message

    headers = _get_headers()

    with Spinner("Cancelling job...") as spinner:
        try:
            response = requests.post(
                f"{ENGINE}/job/cancel/{job_id}",
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to cancel job: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code not in {200, 202}:
            return False, _response_message(
                result, f"Job cancellation failed. Status code: {response.status_code}"
            )

        if not wait:
            return True, _response_message(
                result, f"Job {job_id} cancellation initiated"
            )

        return _poll_job_cancel_until_terminal(job_id, headers, spinner)


def job_list(include_org: bool = False) -> Tuple[bool, str]:
    """
    List jobs - either user's jobs or all org jobs
    Args:
        include_org: If True, list all jobs in the user's organization
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()

    params = {"include_org": include_org} if include_org else {}

    response = requests.get(
        f"{ENGINE}/job/list",
        params=params,
        headers=headers,
        timeout=30,
    )

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error listing jobs. Status code {response.status_code}"
        )
        return False, error_msg

    message = result.get("message")

    return True, message


async def _ws_operation_async(
    endpoint: str,
    spinner: Optional[Spinner] = None,
    payload: Optional[dict] = None,
    handle_user_input: bool = False,
    success_message: str = "Operation completed successfully",
    error_message: str = "Operation failed",
    unexpected_end_message: str = "Connection ended unexpectedly",
) -> Tuple[bool, str]:
    """
    Unified async helper for WebSocket operations (clusters, NFS, etc.)

    Args:
        endpoint: WebSocket endpoint path (e.g., "/cluster/create", "/storage/create")
        spinner: Optional Spinner instance for UI feedback. If None, messages print directly.
        payload: Optional payload to send after API key authentication
        handle_user_input: Whether to handle interactive user input requests
        success_message: Default message for successful completion
        error_message: Default message for errors
        unexpected_end_message: Message when connection ends unexpectedly

    Returns:
        Tuple of (success: bool, message: str)
    """
    api_key = get_tensorpool_key()
    if not api_key:
        return False, "TENSORPOOL_KEY not found. Please set your API key."

    ws_url = (
        f"{ENGINE.replace('http://', 'ws://').replace('https://', 'wss://')}{endpoint}"
    )

    status = None
    msg = None
    close_code = None
    close_reason = None

    try:
        async with websockets.connect(
            ws_url, ping_interval=5, ping_timeout=10
        ) as websocket:
            # First message: Send API key
            await websocket.send(json.dumps({"TENSORPOOL_KEY": api_key}))

            # Second message: Send payload if provided
            if payload is not None:
                await websocket.send(json.dumps(payload))

            # Process server responses
            while True:
                message = await websocket.recv()
                data = json.loads(message)

                status = data.get("status")
                msg = data.get("message")

                if handle_user_input and status == "input":
                    # Server is requesting user input for confirmation
                    # Pause spinner first to clear the line
                    if spinner:
                        spinner.pause()

                    # Print the prompt message directly (don't use spinner.update_text)
                    if msg:
                        user_response = input(msg)
                    else:
                        user_response = input()

                    # Resume spinner after user input
                    if spinner:
                        spinner.resume()

                    # Send user response back to server
                    await websocket.send(json.dumps({"response": user_response}))
                    continue

                if msg:
                    if spinner:
                        spinner.update_text(msg)
                    else:
                        print(msg, flush=True)

                # Break on completion
                if status in ["success", "error"]:
                    break

    except websockets.exceptions.ConnectionClosed as e:
        close_code = e.code
        close_reason = e.reason
        if e.code == 1000:
            # Normal closure
            pass
        elif e.code == 1006:
            # Abnormal closure - connection lost but operation may still be happening
            return (
                False,
                f"Connection lost. The operation may still be in progress.\n{unexpected_end_message}",
            )
        else:
            return (
                False,
                f"Connection closed: code={e.code}, reason={e.reason or 'No reason provided'}",
            )

    except websockets.exceptions.WebSocketException as e:
        return False, f"WebSocket error: {str(e)}"

    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

    if status == "success":
        # If no spinner, msg was already printed to stdout, so return empty string to avoid double-printing
        return True, (msg or success_message) if spinner else ""
    elif status == "error":
        # If no spinner, msg was already printed to stdout, so return empty string to avoid double-printing
        return False, (msg or error_message) if spinner else ""
    else:
        # Connection ended without proper status - include close info if available
        final_message = unexpected_end_message
        if close_code is not None:
            final_message += f"\nClose code: {close_code}"
            if close_reason:
                final_message += f", reason: {close_reason}"
        return False, final_message


def cluster_create(
    identity_file: Optional[str],
    instance_type: str,
    name: Optional[str],
    container: Optional[str],
    num_nodes: Optional[int],
    deletion_protection: bool = False,
    wait: bool = False,
) -> Tuple[bool, str]:
    """
    Create a new cluster (cluster command)
    Args:
        identity_file: Optional path to public SSH key file
        instance_type: Instance type (e.g. 1xH100, 2xH100, 4xH100, 8xH100)
        name: Optional cluster name
        container: Optional container image override
        num_nodes: Number of nodes (must be >= 1)
        deletion_protection: Enable deletion protection for the cluster
        no_input: Whether to skip interactive input prompts
        wait: Whether to wait for the cluster to be fully provisioned before returning
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not instance_type:
        return False, "Instance type is required"
    # num_nodes will be validated server side

    # Build payload
    config_payload = {
        "instance_type": instance_type,
        "num_nodes": num_nodes,
        "deletion_protection": deletion_protection,
    }

    # Only add public_keys if identity_file is provided
    if identity_file:
        # Resolve path and read key
        ssh_key_path = os.path.expanduser(identity_file)
        if not os.path.exists(ssh_key_path):
            return False, f"SSH key file not found: {ssh_key_path}"

        try:
            with open(ssh_key_path, "r") as f:
                ssh_key_content = f.read().strip()
        except Exception as e:
            return False, f"Failed to read SSH key: {e}"

        config_payload["public_keys"] = [ssh_key_content]

    if name:
        config_payload["tp_cluster_name"] = name

    if container:
        config_payload["container"] = container

    headers = _get_headers()

    with Spinner("Creating cluster...") as spinner:
        try:
            response = requests.post(
                f"{ENGINE}/cluster/create",
                json=config_payload,
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to create cluster: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 202:
            return False, _response_message(
                result, f"Cluster creation failed. Status code: {response.status_code}"
            )

        if not wait:
            return True, _response_message(result, "Cluster creation initiated")

        request_id = result.get("request_id")
        if not request_id:
            return True, _response_message(result, "Cluster creation initiated")

        return _poll_request_until_terminal(
            request_id=request_id,
            headers=headers,
            spinner=spinner,
            pending_text="Waiting for cluster provisioning...",
        )


def cluster_destroy(cluster_id: str, no_input: bool = False, wait: bool = False) -> Tuple[bool, str]:
    """
    Destroy a cluster (cluster command)
    Args:
        cluster_id: The ID of the cluster to destroy
        no_input: Whether to skip interactive confirmation prompts
        wait: Whether to wait for the cluster to be fully destroyed before returning
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    confirmed, message = _confirm_destructive_action("Destroy cluster", cluster_id, no_input)
    if not confirmed:
        return False, message

    headers = _get_headers()

    with Spinner("Destroying cluster...") as spinner:
        try:
            response = requests.delete(
                f"{ENGINE}/cluster/{cluster_id}",
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to destroy cluster: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 202:
            return False, _response_message(
                result, f"Cluster destruction failed. Status code: {response.status_code}"
            )

        if not wait:
            return True, _response_message(result, f"Cluster {cluster_id} destruction initiated")

        request_id = result.get("request_id")
        if not request_id:
            return True, _response_message(result, f"Cluster {cluster_id} destruction initiated")

        return _poll_request_until_terminal(
            request_id=request_id,
            headers=headers,
            spinner=spinner,
            pending_text="Waiting for cluster destruction...",
        )


def cluster_list(include_org: bool = False, instances: bool = False) -> Tuple[bool, str]:
    """
    List clusters - either user's clusters or all org clusters
    Args:
        include_org: If True, list all clusters in the user's organization
        instances: If True, show all instances across clusters
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()

    params: dict = {}
    if include_org:
        params["include_org"] = True
    if instances:
        params["instances"] = True

    response = requests.get(
        f"{ENGINE}/cluster/list",
        params=params,
        headers=headers,
        timeout=30,
    )

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error listing clusters. Status code {response.status_code}"
        )
        return False, error_msg

    message = result.get("message")

    return True, message


def cluster_info(cluster_id: str) -> Tuple[bool, str]:
    """
    Get detailed information about a specific cluster
    Args:
        cluster_id: The ID of the cluster to get information about
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not cluster_id:
        return False, "Cluster ID is required"

    headers = _get_headers()

    response = requests.get(
        f"{ENGINE}/cluster/info/{cluster_id}",
        headers=headers,
        timeout=30,
    )

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error getting cluster info. Status code {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "")

    return True, message


def job_info(job_id: str) -> Tuple[bool, str]:
    """
    Get detailed information about a specific job
    Args:
        job_id: The ID of the job to get information about
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not job_id:
        return False, "Job ID is required"

    headers = _get_headers()

    response = requests.get(
        f"{ENGINE}/job/info/{job_id}",
        headers=headers,
        timeout=30,
    )

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error getting job info. Status code {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "")

    return True, message


def ssh_command(
    instance_id: str, ssh_args: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Get SSH command for an instance
    Args:
        instance_id: The ID of the instance to SSH into
        ssh_args: Additional SSH arguments to pass through
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not instance_id:
        return False, "Instance ID is required"

    headers = _get_headers(content_type="")
    # Remove Content-Type for this endpoint
    del headers["Content-Type"]

    try:
        response = requests.get(
            f"{ENGINE}/ssh/{instance_id}",
            headers=headers,
            params={"system": platform.system()},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to get SSH command: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return False, f"Malformed server response. Status code: {response.status_code}"

    if response.status_code != 200:
        message = result.get(
            "message",
            f"Error fetching ssh command. Status code: {response.status_code}\nRun `tp cluster list` to find the instance's info.",
        )
        return False, message

    command = result.get("command")
    message = result.get("message")

    if message:
        print(message)

    if command:
        try:
            command_args = shlex.split(command, posix=(os.name != "nt"))
        except ValueError as e:
            return False, f"Malformed ssh command from server: {str(e)}"

        if not command_args:
            return False, "ssh response did not include an executable command"

        if ssh_args:
            command_args.extend(ssh_args)

        try:
            os.execvpe(command_args[0], command_args, os.environ.copy())
            return True, ""
        except OSError as e:
            return False, f"Failed to execute SSH command: {str(e)}"
    else:
        return False, "ssh response not received from server"


def fetch_user_info() -> Tuple[bool, str]:
    """
    Fetch current user information from the engine
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers(content_type="")
    # Remove Content-Type for this endpoint
    del headers["Content-Type"]

    try:
        response = requests.get(
            f"{ENGINE}/user/info",
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to fetch user info: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return False, f"Malformed server response. Status code: {response.status_code}"

    if response.status_code != 200:
        message = result.get(
            "message",
            f"Error fetching user information. Status code: {response.status_code}",
        )
        return False, message

    message = result.get("message", "")
    return True, message


def storage_create(
    name: Optional[str],
    size: Optional[int],
    storage_type: str,
    deletion_protection: bool = False,
    wait: bool = False,
) -> Tuple[bool, str]:
    """
    Create a new storage volume
    Args:
        name: Optional name for the storage volume
        size: Size of the storage volume in GB
        storage_type: Type of storage volume
        deletion_protection: Enable deletion protection for the storage volume
        wait: Whether to wait for the storage volume to be fully created
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    # Build payload
    payload: dict = {"storage_type": storage_type, "deletion_protection": deletion_protection}
    if size is not None:
        payload["size"] = size
    if name:
        payload["name"] = name

    headers = _get_headers()

    with Spinner("Creating storage volume...") as spinner:
        try:
            response = requests.post(
                f"{ENGINE}/storage/create",
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to create storage volume: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 202:
            return False, _response_message(
                result, f"Storage volume creation failed. Status code: {response.status_code}"
            )

        if not wait:
            return True, _response_message(result, "Storage volume creation initiated")

        request_id = result.get("request_id")
        if not request_id:
            return True, _response_message(result, "Storage volume creation initiated")

        return _poll_request_until_terminal(
            request_id=request_id,
            headers=headers,
            spinner=spinner,
            pending_text="Waiting for storage provisioning...",
        )


def storage_destroy(storage_id: str, no_input: bool = False, wait: bool = False) -> Tuple[bool, str]:
    """
    Destroy a storage volume
    Args:
        storage_id: The ID of the storage volume to destroy
        no_input: Whether to skip interactive confirmation prompts
        wait: Whether to wait for the storage volume to be fully destroyed (unused, kept for API compat)
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "Storage ID is required"

    confirmed, message = _confirm_destructive_action("Destroy storage volume", storage_id, no_input)
    if not confirmed:
        return False, message

    headers = _get_headers()

    with Spinner("Destroying storage volume...") as spinner:
        try:
            response = requests.delete(
                f"{ENGINE}/storage/{storage_id}",
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to destroy storage volume: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 202:
            return False, _response_message(
                result,
                f"Storage volume destruction failed. Status code: {response.status_code}",
            )

        if not wait:
            return True, _response_message(result, f"Storage volume {storage_id} destruction initiated")

        request_id = result.get("request_id")
        if not request_id:
            return True, _response_message(result, f"Storage volume {storage_id} destruction initiated")

        return _poll_request_until_terminal(
            request_id=request_id,
            headers=headers,
            spinner=spinner,
            pending_text="Waiting for storage destruction...",
        )


def storage_attach(
    storage_id: str, cluster_ids: List[str], no_input: bool = False, wait: bool = False
) -> Tuple[bool, str]:
    """
    Attach a storage volume to one or more clusters
    Args:
        storage_id: The ID of the storage volume
        cluster_ids: List of cluster IDs to attach the volume to
        no_input: Whether to skip interactive input prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "No storage ID provided"

    if not cluster_ids:
        return False, "No cluster IDs provided"

    # Build payload
    payload = {"storage_id": storage_id, "cluster_ids": cluster_ids}

    headers = _get_headers()

    with Spinner("Attaching storage volume...") as spinner:
        try:
            response = requests.post(
                f"{ENGINE}/storage/attach",
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to attach storage volume: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 202:
            return False, _response_message(
                result, f"Storage volume attachment failed. Status code: {response.status_code}"
            )

        if not wait:
            return True, _response_message(result, "Storage volume attachment initiated")

        request_ids = result.get("request_ids") or []
        if not request_ids:
            return True, _response_message(result, "Storage volume attachment initiated")

        return _poll_multiple_requests_until_terminal(
            request_ids=request_ids,
            headers=headers,
            spinner=spinner,
            pending_text="Waiting for storage attachment...",
        )


def storage_detach(
    storage_id: str, cluster_ids: List[str], no_input: bool = False, wait: bool = False
) -> Tuple[bool, str]:
    """
    Detach a storage volume from one or more clusters
    Args:
        storage_id: The ID of the storage volume
        cluster_ids: List of cluster IDs to detach the volume from
        no_input: Whether to skip interactive input prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "No storage ID provided"

    if not cluster_ids:
        return False, "No cluster IDs provided"

    if len(cluster_ids) != 1:
        return False, "Exactly one cluster ID is required"

    # Build payload
    payload = {"storage_id": storage_id, "cluster_id": cluster_ids[0]}
    headers = _get_headers()

    with Spinner("Detaching storage volume...") as spinner:
        try:
            response = requests.post(
                f"{ENGINE}/storage/detach",
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            return False, f"Failed to detach storage volume: {str(e)}"

        result, decode_error = _decode_response_json(response)
        if decode_error:
            return False, decode_error

        if response.status_code != 202:
            return False, _response_message(
                result, f"Storage volume detachment failed. Status code: {response.status_code}"
            )

        if not wait:
            return True, _response_message(result, "Storage volume detachment initiated")

        request_id = result.get("request_id")
        if not request_id:
            return True, _response_message(result, "Storage volume detachment initiated")

        return _poll_request_until_terminal(
            request_id=request_id,
            headers=headers,
            spinner=spinner,
            pending_text="Waiting for storage detachment...",
        )


def storage_list(include_org: bool = False) -> Tuple[bool, str]:
    """
    List storage volumes - either user's volumes or all org volumes
    Args:
        include_org: If True, list all volumes in the user's organization
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()

    # Add org parameter to request
    params = {"include_org": include_org} if include_org else {}

    try:
        response = requests.get(
            f"{ENGINE}/storage/list",
            params=params,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to list storage volumes: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return False, f"Malformed server response. Status code: {response.status_code}"

    if response.status_code != 200:
        message = result.get(
            "message", f"Error listing storage volumes. Status code: {response.status_code}"
        )
        return False, message

    message = result.get("message", "")
    return True, message


def storage_info(storage_id: str) -> Tuple[bool, str]:
    """
    Get detailed information about a specific storage volume
    Args:
        storage_id: The ID of the storage volume to get information about
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "Storage ID is required"

    headers = _get_headers()

    try:
        response = requests.get(
            f"{ENGINE}/storage/info/{storage_id}",
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to get storage volume info: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message",
            f"Error getting storage volume info. Status code {response.status_code}",
        )
        return False, error_msg

    message = result.get("message", "")

    return True, message


def cluster_edit(
    cluster_id: str,
    name: Optional[str] = None,
    deletion_protection: Optional[bool] = None,
) -> Tuple[bool, str]:
    """
    Edit cluster properties
    Args:
        cluster_id: The ID of the cluster to edit
        name: Optional new name for the cluster
        deletion_protection: Optional new deletion protection setting
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not cluster_id:
        return False, "Cluster ID is required"

    payload = {}

    if name is not None:
        payload["cluster_name"] = name
    if deletion_protection is not None:
        payload["deletion_protection"] = deletion_protection

    if len(payload) == 0:
        return False, "No properties specified to edit. Provide --name and/or --deletion-protection."

    headers = _get_headers()

    try:
        response = requests.patch(
            f"{ENGINE}/cluster/edit/{cluster_id}",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to edit cluster: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 202:
        error_msg = result.get(
            "message", f"Error editing cluster. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "Cluster edited successfully")
    return True, message


def storage_edit(
    storage_id: str,
    name: Optional[str] = None,
    deletion_protection: Optional[bool] = None,
    size: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Edit storage volume properties
    Args:
        storage_id: The ID of the storage volume to edit
        name: Optional new name for the storage volume
        deletion_protection: Optional new deletion protection setting
        size: Optional new size for the storage volume in GB (can only be increased)
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "Storage ID is required"

    headers = _get_headers()

    payload = {}

    if name is not None:
        payload["tp_storage_name"] = name
    if deletion_protection is not None:
        payload["deletion_protection"] = deletion_protection
    if size is not None:
        payload["size_gb"] = size

    if len(payload) == 0:
        return False, "No properties specified to edit. Provide --name, --deletion-protection, and/or --size."

    try:
        response = requests.patch(
            f"{ENGINE}/storage/edit/{storage_id}",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to edit storage volume: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 202:
        error_msg = result.get(
            "message", f"Error editing storage volume. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "Storage volume edited successfully")
    return True, message


def job_delete(job_id: str, no_input: bool = False) -> Tuple[bool, str]:
    """
    Delete a terminal job (hides it from API endpoints)
    Args:
        job_id: The ID of the job to delete
        no_input: Whether to skip interactive confirmation prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not job_id:
        return False, "Job ID is required"

    headers = _get_headers()

    try:
        response = requests.delete(
            f"{ENGINE}/job/delete/{job_id}",
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to delete job: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error deleting job. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "Job deleted successfully")
    return True, message


def ssh_key_create(key_path: str, name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Create an SSH public key in TensorPool

    Args:
        key_path: Path to the SSH public key file
        name: Optional name for the SSH key

    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not key_path:
        return False, "SSH key path is required"

    # Expand user home directory if needed
    key_path = os.path.expanduser(key_path)

    if not os.path.exists(key_path):
        return False, f"SSH key file not found: {key_path}"

    if not os.path.isfile(key_path):
        return False, f"Path is not a file: {key_path}"

    # Read the public key
    try:
        with open(key_path, "r") as f:
            public_key = f.read().strip()
    except IOError as e:
        return False, f"Failed to read SSH key file: {str(e)}"

    if not public_key:
        return False, "SSH key file is empty"

    headers = _get_headers()
    payload = {"public_key": public_key}

    if name:
        payload["name"] = name

    try:
        response = requests.post(
            f"{ENGINE}/user/ssh-key/add",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to add SSH key: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error adding SSH key. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "SSH key added successfully")
    return True, message


def ssh_key_list(include_org: bool = False) -> Tuple[bool, str]:
    """
    List all SSH keys registered with TensorPool

    Args:
        include_org: If True, list all SSH keys in the organization

    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()
    params = {"include_org": include_org} if include_org else {}

    try:
        response = requests.get(
            f"{ENGINE}/user/ssh-key/list",
            headers=headers,
            params=params,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to list SSH keys: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error listing SSH keys. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "")
    return True, message


def ssh_key_destroy(key_id: str) -> Tuple[bool, str]:
    """
    Remove an SSH key from TensorPool

    Args:
        key_id: The ID of the SSH key to remove

    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not key_id:
        return False, "SSH key ID is required"

    headers = _get_headers()

    try:
        response = requests.delete(
            f"{ENGINE}/user/ssh-key/remove/{key_id}",
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to remove SSH key: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error removing SSH key. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "SSH key removed successfully")
    return True, message
