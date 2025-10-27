import os
import time
from typing import Final, Optional, List, Dict, Tuple
import requests
from tqdm import tqdm
import importlib.metadata
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import asyncio
import websockets
import threading
import queue
from .spinner import Spinner
import platform

ENGINE: Final = os.environ.get("TENSORPOOL_ENGINE", "https://engine.tensorpool.dev")


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


def fetch_dashboard() -> str:
    """
    Fetch the TensorPool dashboard URL
    """

    timezone = time.strftime("%z")
    # print(timezone)

    headers = _get_headers()
    payload = {
        "timezone": timezone,  # Timezone to formate timestamps
    }

    fallback_dashboard_msg = "https://tensorpool.dev/dashboard"

    try:
        response = requests.post(
            f"{ENGINE}/dashboard",
            json=payload,
            headers=headers,
            timeout=15,
        )

        try:
            res = response.json()
        except requests.exceptions.JSONDecodeError:
            return fallback_dashboard_msg

        message = res.get("message", fallback_dashboard_msg)
        return message

    except Exception as e:
        raise RuntimeError(f"Failed to fetch dashboard URL: {str(e)}")


async def _job_push_async(
    tp_config: str,
    public_key_contents: str,
    api_key: str,
    tensorpool_pub_key_path: str,
    tensorpool_priv_key_path: str,
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
                "public_key_path": tensorpool_pub_key_path,
                "private_key_path": tensorpool_priv_key_path,
                "public_keys": [public_key_contents],
                "system": platform.system(),
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
                    # print("command:",command)
                    # print("show_stdout:", show_stdout)

                    try:
                        # Set up environment to force unbuffered output
                        env = os.environ.copy()
                        env["PYTHONUNBUFFERED"] = "1"

                        process = subprocess.Popen(
                            command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL,  # Don't accept stdin
                            text=True,
                            bufsize=1,  # Line buffering
                            universal_newlines=True,
                            env=env,
                        )

                        # Use threading to read stdout and stderr concurrently
                        stdout_lines = []
                        stderr_lines = []
                        stdout_queue = queue.Queue()
                        stderr_queue = queue.Queue()

                        def read_stdout():
                            while True:
                                line = process.stdout.readline()
                                if not line:
                                    break
                                stdout_queue.put(line)
                                stdout_lines.append(line)
                                if show_stdout:
                                    sys.stdout.write(line)
                                    sys.stdout.flush()

                        def read_stderr():
                            while True:
                                line = process.stderr.readline()
                                if not line:
                                    break
                                stderr_queue.put(line)
                                stderr_lines.append(line)
                                if (
                                    show_stdout
                                ):  # Show stderr in real-time when show_stdout is True
                                    sys.stderr.write(line)
                                    sys.stderr.flush()

                        # Start threads to read stdout and stderr
                        stdout_thread = threading.Thread(target=read_stdout)
                        stderr_thread = threading.Thread(target=read_stderr)
                        stdout_thread.daemon = True
                        stderr_thread.daemon = True
                        stdout_thread.start()
                        stderr_thread.start()

                        # Wait for process completion
                        returncode = process.wait()

                        # Wait for threads to finish reading
                        stdout_thread.join(timeout=1)
                        stderr_thread.join(timeout=1)

                        stdout = "".join(stdout_lines)
                        stderr = "".join(stderr_lines)

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
    tensorpool_pub_key_path: str,
    tensorpool_priv_key_path: str,
) -> Tuple[bool, Optional[str]]:
    """
    Push a job
    Args:
        tp_config_path: Path to the tp config file
        tensorpool_pub_key_path: Path to tensorpool public key
        tensorpool_priv_key_path: Path to tensorpool private key
    Returns:
        Tuple[bool, Optional[str]]: (success status, job_id if available)
    """
    if not os.path.exists(tp_config_path):
        print(f"Config file not found: {tp_config_path}")
        return False, None

    # Check that both key paths are provided
    if not tensorpool_pub_key_path or not tensorpool_priv_key_path:
        print("Both tensorpool public and private key paths are required")
        return False, None

    if not os.path.exists(tensorpool_pub_key_path):
        print(f"Public key file not found: {tensorpool_pub_key_path}")
        return False, None
    if not os.path.exists(tensorpool_priv_key_path):
        print(f"Private key file not found: {tensorpool_priv_key_path}")
        return False, None

    try:
        with open(tp_config_path, "r") as f:
            tp_config = f.read()
    except Exception as e:
        print(f"Failed to read {tp_config_path}: {str(e)}")
        return False, None

    try:
        with open(tensorpool_pub_key_path, "r") as f:
            public_key_contents = f.read().strip()
    except Exception as e:
        print(f"Failed to read {tensorpool_pub_key_path}: {str(e)}")
        return False, None

    api_key = get_tensorpool_key()
    if not api_key:
        print("TENSORPOOL_KEY not found. Please set your API key.")
        return False, None

    # Run the async function
    return asyncio.run(
        _job_push_async(
            tp_config,
            public_key_contents,
            api_key,
            tensorpool_pub_key_path,
            tensorpool_priv_key_path,
        )
    )


def job_pull(
    job_id: str,
    files: Optional[List[str]] = None,
    dry_run: bool = False,
    tensorpool_priv_key_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Pull job output files or execute commands
    Args:
        job_id: The ID of the job to pull
        files: Optional list of specific files to pull
        dry_run: If True, only preview files without downloading
        tensorpool_priv_key_path: Path to tensorpool private key
    Returns:
        A tuple containing a download map (or None) and an optional message
    """
    if not job_id:
        return None, "Job ID is required"

    headers = _get_headers()
    params = {
        "dry_run": dry_run,
        "system": platform.system(),
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
                # Set up environment to force unbuffered output
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                )

                # Read output in real-time
                stdout_lines = []
                stderr_lines = []

                # Read stdout line by line
                for line in process.stdout:
                    stdout_lines.append(line)
                    if show_stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()

                # Wait for process to complete and get any remaining stderr
                stderr = process.stderr.read()
                if stderr:
                    stderr_lines.append(stderr)
                    if show_stdout:
                        sys.stderr.write(stderr)
                        sys.stderr.flush()

                returncode = process.wait()

                # Print errors if command failed and not already shown
                if returncode != 0 and not show_stdout:
                    stderr_text = "".join(stderr_lines)
                    stdout_text = "".join(stdout_lines)
                    if stderr_text:
                        print(f"Command error: {stderr_text}")
                    if stdout_text:
                        print(f"Command output: {stdout_text}")

            except Exception as e:
                print(f"Failed to execute command: {str(e)}")
                return None, f"Failed to execute command: {str(e)}"

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

                    if response.status_code == 200:
                        return True, (file_path, response.status_code, "Success")

                    if retries < max_retries:
                        delay = base_delay * (2**retries)  # Exponential backoff
                        time.sleep(delay)
                        continue

                    return False, (file_path, response.status_code, response.text)

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


def job_cancel(job_id: str, no_input: bool = False) -> Tuple[bool, str]:
    """
    Cancel a job
    Args:
        job_id: The ID of the job to cancel
        no_input: Whether to skip interactive confirmation prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    assert job_id is not None, "A job ID is needed to cancel"

    # Build endpoint
    endpoint = f"/job/cancel/{job_id}"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Cancelling job...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=None,
                handle_user_input=not no_input,
                success_message=f"Job {job_id} cancelled successfully",
                error_message="Job cancellation failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during job cancellation.\n"
                    "The job may still be cancelling. Check 'tp job list' to see the current status."
                ),
            )
        )

    return success, message


def job_list(org: bool = False) -> Tuple[bool, str]:
    """
    List jobs - either user's jobs or all org jobs
    Args:
        org: If True, list all jobs in the user's organization
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()

    params = {"org": org} if org else {}

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
        endpoint: WebSocket endpoint path (e.g., "/cluster/create", "/nfs/create")
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
    num_nodes: Optional[int],
    deletion_protection: bool = False,
    no_input: bool = False,
) -> Tuple[bool, str]:
    """
    Create a new cluster (cluster command)
    Args:
        identity_file: Optional path to public SSH key file
        instance_type: Instance type (e.g. 1xH100, 2xH100, 4xH100, 8xH100)
        name: Optional cluster name
        num_nodes: Number of nodes (must be >= 1)
        deletion_protection: Enable deletion protection for the cluster
        no_input: Whether to skip interactive input prompts
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

    # Build endpoint
    endpoint = "/cluster/create"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Creating cluster...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=config_payload,
                handle_user_input=not no_input,
                success_message="Cluster created successfully",
                error_message="Cluster creation failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during cluster creation.\n"
                    "The cluster may still be provisioning. Check 'tp cluster list' to see the current status."
                ),
            )
        )

    return success, message


def cluster_destroy(cluster_id: str, no_input: bool = False) -> Tuple[bool, str]:
    """
    Destroy a cluster (cluster command)
    Args:
        cluster_id: The ID of the cluster to destroy
        no_input: Whether to skip interactive confirmation prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    # Build endpoint
    endpoint = f"/cluster/destroy/{cluster_id}"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Destroying cluster...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=None,
                handle_user_input=not no_input,
                success_message=f"Cluster {cluster_id} destroyed successfully",
                error_message="Cluster destruction failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during cluster destruction.\n"
                    "The cluster may still be destroying. Check 'tp cluster list' to see the current status."
                ),
            )
        )

    return success, message


def cluster_list(org: bool = False) -> Tuple[bool, str]:
    """
    List clusters - either user's clusters or all org clusters
    Args:
        org: If True, list all clusters in the user's organization
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()

    params = {"org": org} if org else {}

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
            f"{ENGINE}/ssh/connect/{instance_id}",
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
        # Execute the SSH command interactively
        try:
            # Append additional SSH arguments if provided
            if ssh_args:
                additional_args = " ".join(ssh_args)
                full_command = f"{command} {additional_args}"
            else:
                full_command = command

            subprocess.run(full_command, shell=True)
            return True, ""
        except KeyboardInterrupt:
            return True, "\nSSH session terminated"
        except Exception as e:
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
            f"{ENGINE}/me",
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


def nfs_create(
    name: Optional[str],
    size: int,
    deletion_protection: bool = False,
    no_input: bool = False,
) -> Tuple[bool, str]:
    """
    Create a new NFS volume
    Args:
        name: Optional name for the NFS volume
        size: Size of the NFS volume in GB
        deletion_protection: Enable deletion protection for the NFS volume
        no_input: Whether to skip interactive input prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    # Build payload
    payload = {"size": size, "deletion_protection": deletion_protection}
    if name:
        payload["name"] = name

    # Build endpoint
    endpoint = "/nfs/create"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Creating NFS volume...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=payload,
                handle_user_input=not no_input,
                success_message="NFS volume created successfully",
                error_message="NFS volume creation failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during NFS volume creation.\n"
                    "The volume may still be provisioning. Check 'tp nfs list' to see the current status."
                ),
            )
        )

    return success, message


def nfs_destroy(storage_id: str, no_input: bool = False) -> Tuple[bool, str]:
    """
    Destroy an NFS volume
    Args:
        storage_id: The ID of the NFS volume to destroy
        no_input: Whether to skip interactive confirmation prompts
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "Storage ID is required"

    # Build endpoint
    endpoint = f"/nfs/destroy/{storage_id}"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Destroying NFS volume...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=None,
                handle_user_input=not no_input,
                success_message=f"NFS volume {storage_id} destroyed successfully",
                error_message="NFS volume destruction failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during NFS volume destruction.\n"
                    "The volume may still be destroying. Check 'tp nfs list' to see the current status."
                ),
            )
        )

    return success, message


def nfs_attach(
    storage_id: str, cluster_ids: List[str], no_input: bool = False
) -> Tuple[bool, str]:
    """
    Attach an NFS volume to one or more clusters
    Args:
        storage_id: The ID of the NFS volume
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

    # Build endpoint
    endpoint = "/nfs/attach"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Attaching NFS volume...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=payload,
                handle_user_input=not no_input,
                success_message="NFS volume attached successfully",
                error_message="NFS volume attachment failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during NFS volume attachment.\n"
                    "The attachment may still be in progress. Check 'tp nfs list' to see the current status."
                ),
            )
        )

    return success, message


def nfs_detach(
    storage_id: str, cluster_ids: List[str], no_input: bool = False
) -> Tuple[bool, str]:
    """
    Detach an NFS volume from one or more clusters
    Args:
        storage_id: The ID of the NFS volume
        cluster_ids: List of cluster IDs to detach the volume from
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

    # Build endpoint
    endpoint = "/nfs/detach"
    if no_input:
        endpoint += "?no_input=true"

    # Run the async function with a spinner
    with Spinner("Detaching NFS volume...") as spinner:
        success, message = asyncio.run(
            _ws_operation_async(
                endpoint=endpoint,
                spinner=spinner,
                payload=payload,
                handle_user_input=not no_input,
                success_message="NFS volume detached successfully",
                error_message="NFS volume detachment failed",
                unexpected_end_message=(
                    "Connection ended unexpectedly during NFS volume detachment.\n"
                    "The detachment may still be in progress. Check 'tp nfs list' to see the current status."
                ),
            )
        )

    return success, message


def nfs_list(org: bool = False) -> Tuple[bool, str]:
    """
    List NFS volumes - either user's volumes or all org volumes
    Args:
        org: If True, list all volumes in the user's organization
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()

    # Add org parameter to request
    params = {"org": org} if org else {}

    try:
        response = requests.get(
            f"{ENGINE}/nfs/list",
            params=params,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to list NFS volumes: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return False, f"Malformed server response. Status code: {response.status_code}"

    if response.status_code != 200:
        message = result.get(
            "message", f"Error listing NFS volumes. Status code: {response.status_code}"
        )
        return False, message

    message = result.get("message", "")
    return True, message


def nfs_info(storage_id: str) -> Tuple[bool, str]:
    """
    Get detailed information about a specific NFS volume
    Args:
        storage_id: The ID of the NFS volume to get information about
    Returns:
        A tuple containing a boolean indicating success and a message
    """
    if not storage_id:
        return False, "Storage ID is required"

    headers = _get_headers()

    try:
        response = requests.get(
            f"{ENGINE}/nfs/info/{storage_id}",
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to get NFS volume info: {str(e)}"

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
            f"Error getting NFS volume info. Status code {response.status_code}",
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

    headers = _get_headers()

    payload = {}

    if name is not None:
        payload["tp_cluster_name"] = name
    if deletion_protection is not None:
        payload["deletion_protection"] = deletion_protection

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

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error editing cluster. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "Cluster edited successfully")
    return True, message


def nfs_edit(
    storage_id: str,
    name: Optional[str] = None,
    deletion_protection: Optional[bool] = None,
    size: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Edit NFS volume properties
    Args:
        storage_id: The ID of the NFS volume to edit
        name: Optional new name for the NFS volume
        deletion_protection: Optional new deletion protection setting
        size: Optional new size for the NFS volume in GB (can only be increased)
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
        return False, "No properties specified to edit"

    try:
        response = requests.patch(
            f"{ENGINE}/nfs/edit/{storage_id}",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Failed to edit NFS volume: {str(e)}"

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            f"Failed to decode server response. Status code: {response.status_code}",
        )

    if response.status_code != 200:
        error_msg = result.get(
            "message", f"Error editing NFS volume. Status code: {response.status_code}"
        )
        return False, error_msg

    message = result.get("message", "NFS volume edited successfully")
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
            f"{ENGINE}/ssh/key/create",
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


def ssh_key_list(org: bool = False) -> Tuple[bool, str]:
    """
    List all SSH keys registered with TensorPool

    Args:
        org: If True, list all SSH keys in the organization

    Returns:
        A tuple containing a boolean indicating success and a message
    """
    headers = _get_headers()
    params = {"org": org}

    try:
        response = requests.get(
            f"{ENGINE}/ssh/key/list",
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
            f"{ENGINE}/ssh/key/destroy/{key_id}",
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
