import os
import tarfile
import subprocess
from typing import Final, Optional, List, Dict
import requests
from tqdm import tqdm
import toml
from toml.encoder import TomlEncoder
import importlib.metadata

# ENGINE: Final = "http://localhost:8000"
ENGINE: Final = "https://engine.tensorpool.dev"

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
            timeout=10,
        )
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            # Malformed response handling
            # print(response.text)
            return (
                False,
                f"Received malformed response from server. Status code: {response.status_code} \nIf this persists, please contact team@tensorpool.dev",
            )
        if response.status_code == 200:
            # Valid health
            return (True, data["message"])
        else:
            # Engine online, but auth or health check failure
            return (False, data["message"])
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
    files = [
        os.path.join(dirpath, f)
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
    """
    contents = ""

    assert os.path.exists(file_path), f"File not found: {file_path}"
    with open(file_path, "r") as f:
        contents += f.read()
    return contents.strip()

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
    assert len(file_paths) > 0, "No files found in the project directory, are you in the right directory?"

    CTX_IGNORE_SUFFIXES = {"tp-config.toml"}
    filtered_paths = [f for f in file_paths if is_utf8_encoded(f) and not any(f.endswith(i) for i in CTX_IGNORE_SUFFIXES)]

    filtered_file_contents: Dict[str, str] = {f: get_file_contents(f) for f in filtered_paths}

    return file_paths, filtered_file_contents

def create_proj_tarball(tmp_dir: str) -> str:
    """
    Create a tarball of the project directory.
    """
    cwd = os.getcwd()
    tarball_path = os.path.join(tmp_dir, "proj.tgz")
    tarball_name = os.path.basename(tarball_path)

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            for item in os.listdir(cwd):
                if item == tarball_name:
                    continue  # Don't add tarball to itself
                elif any(item.endswith(i) for i in IGNORE_FILE_SUFFIXES):
                    continue
                item_path = os.path.join(cwd, item)
                tar.add(item_path, arcname=item)

        assert os.path.exists(tarball_path), "Tarball was not created successfully"

    except Exception as e:
        print(f"Error creating tarball: {str(e)}")
        raise

    return tarball_path


def upload_with_progress(file_path, signed_url):
    file_size = os.path.getsize(file_path)
    headers = {"Content-Type": "application/x-tar"}

    with open(file_path, "rb") as f:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Uploading"
        ) as progress_bar:

            def _data_gen():
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
                    progress_bar.update(len(chunk))

            response = requests.put(signed_url, data=_data_gen(), headers=headers)

    if response.status_code == 200:
        return True
    else:
        print(
            f"\nUpload failed with status code {response.status_code}: {response.text}"
        )
        return False

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


def dump_tp_toml(json: Dict) -> str:
    """
    Convert a JSON object to a TOML string and dumps to tp-config.toml in the cwd
    Arguments:
        json: The Dict to convert to TOML
    Returns:
        The filepath to the created TOML file
    """

    with open("tp-config.toml", "w") as f:
        toml.dump(json, f, encoder=TomlNewlineArrayEncoder())

    tp_config_path = os.path.join(os.getcwd(), "tp-config.toml")

    if not os.path.exists(tp_config_path):
        raise FileNotFoundError("tp-config.toml was not created successfully")

    return tp_config_path

def gen_job_metadata() -> Dict:
    """
    Generate metadata for a job
    """
    headers = {"Content-Type": "application/json"}
    payload = {"key": os.environ["TENSORPOOL_KEY"]}

    try:
        response = requests.post(
            f"{ENGINE}/job/gen", json=payload, headers=headers, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Job metadata generation failed: {str(e)}")


def translate_job(query: str, dir_ctx: List[str], file_ctx: Dict[str, str]) -> Dict:
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
            timeout=60, # limit is quite high...
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(e)
        # raise RuntimeError(f"Job translation failed: {str(e)}\nPlease try again or create your tp-config.toml manually.")
        raise RuntimeError(
            "Job translation failed. Please try again or create your tp-config.toml manually."
        )


def submit_job(res: Dict, tp_config: Dict) -> str:
    """
    Submit a job to the /submit-job endpoint
    Args:
        res: The translation response (need for job id)
        tp_config: The user defined TensorPool configuration
    Output:
        The link to the job
    """

    merged = {**res, **tp_config}
    merged.pop("upload_url", None)  # Don't need the upload url at this point
    assert merged["is_valid_job"], "Job is not valid, we should not be here.."
    merged.pop("is_valid_job", None)

    headers = {"Content-Type": "application/json"}
    payload = {"key": os.environ["TENSORPOOL_KEY"], "tp-config": merged}

    try:
        response = requests.post(
            f"{ENGINE}/job/submit", json=payload, headers=headers, timeout=30
        )
        res = response.json()

        if res["status"] == "success":
            return res["link"]
        else:
            print("Job submission failed.")
            print(res["message"])

    except Exception as e:
        raise RuntimeError(f"Job submission failed: {str(e)}\nPlease try again.")
