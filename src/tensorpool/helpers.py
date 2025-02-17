import os
import time
from typing import Final, Optional, List, Dict, Tuple
import requests
from tqdm import tqdm
import toml
from toml.encoder import TomlEncoder
import importlib.metadata
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed


# ENGINE: Final = "http://localhost:8000"
ENGINE: Final = "https://engine.tensorpool.dev/"

# TODO: deprecate, should all be in tpignore
IGNORE_FILE_SUFFIXES: Final = {
    "venv",
    "DS_Store",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
}


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


def login():
    """
    Store the API key in the .env file and set it in the environment variables.
    """
    print("https://tensorpool.dev/dashboard")
    api_key = input("Enter your TensorPool API key: ").strip()

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

    key = os.getenv("TENSORPOOL_KEY")
    try:
        version = get_version()
        # print(f"Package version: {version}")
        response = requests.post(
            f"{ENGINE}/health",
            json={"key": key, "package_version": version},
            timeout=15,
        )
        # TODO: status code before res parse? if res json parse fails. do for other fn too
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            # Malformed response handling
            # print(response.text)
            return (
                False,
                f"Received malformed response from server. Status code: {response.status_code} \nIf this persists, please contact team@tensorpool.dev",
            )

        msg = data.get("message")

        if response.status_code == 200:
            # Valid health
            return (True, msg)
        else:
            # Engine online, but auth or health check failure
            return (False, msg)
    except requests.exceptions.ConnectionError as e:
        return (
            False,
            "Cannot reach the TensorPool engine. Please check your internet connection.\nHaving trouble? Contact team@tensorpool.dev",
        )
    except Exception as e:
        # Catch-all for unexpected failures
        return (False, f"Unexpected error during health check: {str(e)}")


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


def get_file_contents(file_path: str) -> str:
    """
    Returns the contents of the file
    Args:
        file_path: The path to the file
    Returns:
        The contents of the file
    Raises:
        FileNotFoundError: If the file is not found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.read()
        return contents.strip()
    except UnicodeDecodeError:
        # Fallback to read in binary mode and attempt to decode with error handling
        # TODO: test this
        with open(file_path, "rb") as f:
            raw_contents = f.read()
        return raw_contents.decode("utf-8", errors="replace").strip()


def is_utf8_encoded(file_path: str) -> bool:
    """
    Check if a file is UTF-8 encoded
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def construct_proj_ctx(file_paths: List[str]) -> (List[str], Dict[str, str]):
    """
    Constructs the project context
    Args:
        file_paths: The paths to the files in the project
    Returns:
        A tuple containing the a list of the project directory files and a dict of the file contents
    """
    assert len(file_paths) > 0, (
        "No files found in the project directory, are you in the right directory?"
    )

    filtered_paths = [
        f
        for f in file_paths
        if is_utf8_encoded(f) and not any(f.endswith(i) for i in IGNORE_FILE_SUFFIXES)
    ]

    filtered_file_contents: Dict[str, str] = {
        f: get_file_contents(f) for f in filtered_paths
    }

    return file_paths, filtered_file_contents


# TODO: ignore from tp config


# def should_ignore(path: str, ignore_patterns: set[str]) -> bool:
#     """
#     Check if path matches any ignore patterns.
#     """
#     path = Path(path)
#     name = path.name

#     for pattern in ignore_patterns:
#         # Handle both file/dir names and full paths
#         if fnmatch.fnmatch(name, pattern):
#             return True
#         if fnmatch.fnmatch(str(path), pattern):
#             return True
#     return False


def load_tp_config(path: str) -> Optional[Dict]:
    """
    Load a tp config from a file
    """
    assert os.path.exists(path), f"File not found: {path}"
    config = toml.load(path)
    return config


class TomlNewlineArrayEncoder(TomlEncoder):
    def __init__(self, _dict=dict, preserve=False):
        super(TomlNewlineArrayEncoder, self).__init__(_dict, preserve)

    def dump_list(self, v):
        items = [self.dump_value(item) for item in v]
        # multiline array
        retval = "[\n"
        retval += ",\n".join("  " + x for x in items)
        retval += "\n]"
        return retval


def dump_tp_toml(json: Dict, path: str) -> bool:
    """
    Convert a tp config dict to a TOML string and dump to path
    Arguments:
        json: The Dict to convert to TOML
        path: The path to save the TOML file
    Returns:
        A boolean indicating success
    """

    with open(path, "w") as f:
        toml.dump(json, f, encoder=TomlNewlineArrayEncoder())

    if not os.path.exists(path):
        return False

    return True


def job_init(
    tp_config, project_state_snapshot, skip_cache=False
) -> Tuple[str, str, Dict]:
    """
    Initialize a job
    Returns:
        Tuple of an optional message, job id, and upload map
    """
    assert tp_config is not None, "A TP config must be provided to initialize a job"
    assert project_state_snapshot is not None, (
        "A project state snapshot must be provided to initialize a job"
    )

    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
        "tp-config": tp_config,
        "snapshot": project_state_snapshot,
        "skip_cache": skip_cache,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{ENGINE}/job/init",
            json=payload,
            headers=headers,
            timeout=60,  # snapshot's can be huge...
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Job initialization failed: {str(e)}")

    if response.status_code != 200:
        # print(response.text)
        raise RuntimeError(
            f"TensorPool engine returned status {response.status_code}: {response.text}"
        )

    try:
        res = response.json()
    except requests.exceptions.JSONDecodeError:
        print(response.text)
        raise RuntimeError(
            "Received malformed response from server. Please contact team@tensorpool.dev"
        )

    status = res.get("status")
    id = res.get("id")
    message = res.get("message")
    upload_map = res.get("upload_map")

    if response.status_code != 200 or status != "success":
        return status, message, None

    assert id is not None, "No job ID recieved, please contact team@tensorpool.dev"

    assert upload_map is not None, (
        "No upload map recieved, please contact team@tensporpool.dev"
    )

    return message, id, upload_map


def _upload_single_file(
    file: str, upload_details: dict, pbar: tqdm, max_retries: int = 3
) -> bool:
    """
    Upload a single file with progress tracking.
    Returns (success, bytes_uploaded)
    """
    backoff = 1
    file_size = os.path.getsize(file)

    for attempt in range(1, max_retries + 1):
        try:
            with open(file, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    upload_details["url"], data=upload_details["fields"], files=files
                )

            if response.status_code == 204:
                pbar.write(f"Uploaded {file}")
                pbar.update(file_size)
                return True
            else:
                if attempt == max_retries:
                    pbar.write(f"Failed upload {file}: {response.status_code}")

        except Exception as exc:
            if attempt == max_retries:
                pbar
                pbar.write(f"Exception uploading {file}: {exc}")

        time_to_wait = backoff * (2 ** (attempt - 1))
        time.sleep(time_to_wait)

    return False


def upload_files(upload_map: Dict[str, dict]) -> bool:
    """
    Upload files with a single combined progress bar.
    """
    total_bytes = sum(os.path.getsize(f) for f in upload_map.keys())
    max_workers = min(os.cpu_count() * 2 if os.cpu_count() else 6, 6)

    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"Uploading {len(upload_map)} files",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_upload_single_file, file, upload_details, pbar): file
                for file, upload_details in upload_map.items()
            }

            success = all(future.result() for future in as_completed(futures))

    return success


def job_submit(job_id: str, tp_config: Dict) -> Tuple[bool, str]:
    headers = {"Content-Type": "application/json"}
    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
        "id": job_id,
        "tp-config": tp_config,
    }

    try:
        response = requests.post(
            f"{ENGINE}/job/submit",
            json=payload,
            headers=headers,
            timeout=60,
        )
    except Exception as e:
        raise RuntimeError(f"Job submission failed: {str(e)}\nPlease try again.")

    try:
        res = response.json()
    except requests.exceptions.JSONDecodeError:
        print(response.text)
        raise RuntimeError(
            "Recieved malformed response from server. Please contact team@tensorpool.dev"
        )

    res_status = res.get("status", None)
    message = res.get("message", None)

    if res_status == "success":
        return True, message
    else:
        return False, f"Job submission failed\n{message}"


def autogen_job_config(
    query: str, dir_ctx: List[str], file_ctx: Dict[str, str]
) -> Dict:
    """
    Send query to translate endpoint
    Args:
        query: The query to translate to a task
        dir_ctx: The directory context (all files in the project directory)
        file_ctx: The file context (files and their contents)
    """
    assert query is not None and query != "", "Query cannot be None or empty"
    assert file_ctx is not None and file_ctx != "", (
        "File context cannot be None or empty"
    )
    assert dir_ctx is not None and dir_ctx != "", (
        "Directory context cannot be None or empty"
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
        "query": query,
        "dir_ctx": dir_ctx,
        "file_ctx": file_ctx,
    }
    # print("Payload:", payload)

    # TODO: better capture failed translation
    # TODO: check if proj too large

    try:
        response = requests.post(
            f"{ENGINE}/translate",
            json=payload,
            headers=headers,
            timeout=60,  # limit is quite high...
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(e)
        raise RuntimeError(
            "tp-config.toml autogeneration failed.\nPlease try again or create your tp-config.toml manually."
        )


def save_empty_tp_config(path: str) -> bool:
    """
    Fetch the default empty tp config and save it
    """

    # Gets the default empty tp config
    response = requests.get(f"{ENGINE}/empty-tp-config")

    if response.status_code != 200:
        return False

    with open(path, "w+") as f:
        f.write(response.text)

    return True


def autogen_tp_config(
    query: str, dir_ctx: List[str], file_ctx: Dict[str, str]
) -> Tuple[bool, Dict, Optional[str]]:
    """
    Send query to translate endpoint
    Args:
        query: The query to translate to a task
        dir_ctx: The directory context (all files in the project directory)
        file_ctx: The file context (files and their contents)
    Returns:
        A tuple containing a boolean indicating success, the translated config, and an optional message
    """
    assert query is not None and query != "", "Query cannot be None or empty"
    assert file_ctx is not None and file_ctx != "", (
        "File context cannot be None or empty"
    )
    assert dir_ctx is not None and dir_ctx != "", (
        "Directory context cannot be None or empty"
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
        "query": query,
        "dir_ctx": dir_ctx,
        "file_ctx": file_ctx,
    }
    # print("Payload:", payload)

    # TODO: better capture failed translation
    # TODO: check if proj too large

    try:
        response = requests.post(
            f"{ENGINE}/tp-config-autogen",
            json=payload,
            headers=headers,
            timeout=60,  # limit is quite high...
        )
    except Exception as e:
        return False, {}, str(e)

    try:
        res = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            False,
            {},
            "Malformed response from server. Please contact team@tensorpool.dev",
        )

    response_status = res.get("status", None)
    tp_config = res.get("tp-config", {})
    message = res.get("message", None)

    if response_status != "success":
        return False, tp_config, message
    if tp_config == {}:
        return False, tp_config, message

    return True, tp_config, message

    # print(e)
    # raise RuntimeError(
    #     "tp-config.toml autogeneration failed.\nPlease try again or create your tp-config.toml manually."
    # )


def snapshot_proj_state() -> Dict[str, int]:
    """
    Local state snapshot of the project directory.
    Returns a dictionary all paths and their last modified timestamps
    """
    files = get_proj_paths()
    # TODO: consider if OSError is thrown on getmtime
    return {f: os.path.getmtime(f) for f in files}


def listen_to_job(job_id: str) -> None:
    """Connects to job stream and prints output in real-time."""
    headers = {"Content-Type": "application/json"}
    payload = {"key": os.environ["TENSORPOOL_KEY"], "id": job_id}

    # TODO: pull in stdout once job is succeeded

    try:
        response = requests.post(
            f"{ENGINE}/job/listen", json=payload, headers=headers, stream=True
        )

        if response.status_code != 200:
            print(f"Failed to connect to job stream: {response.text}")
            return

        for line in response.iter_lines():
            if not line:
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                continue

            if text.startswith("data: "):
                pretty = text.replace("data: ", "", 1)
                print(pretty, flush=True)

    except KeyboardInterrupt:
        print("\nDetached from job stream")
    except Exception as e:
        print(f"Error while listening to job stream: {str(e)}")


def fetch_dashboard() -> str:
    """
    Fetch the TensorPool dashboard URL
    """

    timezone = time.strftime("%z")
    # print(timezone)

    headers = {"Content-Type": "application/json"}
    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
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


def job_pull(
    job_id: Optional[str], snapshot: Optional[dict] = None
) -> Tuple[Dict[str, str], str]:
    """
    Given a job ID, fetch the job's output files that changed during the job.
    Returns a download map and a message.
    """

    assert job_id or snapshot, "A job ID or snapshot are needed to pull a job"

    headers = {"Content-Type": "application/json"}
    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
        # can accept either a job ID or a snapshot
        # if only a snapshot is provided, the job id will be inferred
        "snapshot": snapshot,
        "id": job_id,
    }

    try:
        response = requests.post(
            f"{ENGINE}/job/pull", json=payload, headers=headers, timeout=15
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Job pull failed: {str(e)}")

    try:
        res = response.json()
    except requests.exceptions.JSONDecodeError:
        # print(response.text)
        raise RuntimeError(
            "Malformed response from server while pulling job\nPlease try again or visit https://app.tensorpool.dev/dashboard\nContact team@tensorpool.dev if this persists"
        )

    status = res.get("status")
    msg = res.get("message")
    if status != "success":
        return None, msg

    download_map = res.get("download_map")
    return download_map, msg


def download_files(download_map: Dict[str, str]) -> bool:
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
                        base, ext = os.path.splitext(file_path)
                        counter = 1
                        while os.path.exists(f"{base}_{counter}{ext}"):
                            counter += 1
                        file_path = f"{base}_{counter}{ext}"

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
