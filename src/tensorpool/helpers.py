import os
import tarfile
import subprocess
from typing import Final, Optional, List, Dict
import requests
from tqdm import tqdm
import toml
import importlib.metadata

# ENGINE: Final = "http://localhost:8000"
ENGINE: Final = "https://engine.tensorpool.dev"

IGNORE_FILE_SUFFIXES: Final = {"venv", ".git", "DS_Store", "__pycache__", ".idea", ".vscode", "node_modules"}

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
            timeout=5,
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


def get_file_paths():
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


def count_tokens(s: str) -> int:
    """
    Estimates the number of tokens in a string.
    """
    return len(s) // 4


def create_tp_reqs(
    tmp_dir: str, req_output_filename: Optional[str] = "tp-requirements.txt"
) -> str:
    """
    Creates a reproducible requirements file by ensuring all packages and their
    exact versions are on PyPI, without relying on the experimental 'pip index'.
    """
    output_file = os.path.join(tmp_dir, req_output_filename)
    temp_req_in = os.path.join(tmp_dir, "requirements.in")

    # Prepare a list for validated requirements pinned to exact versions
    validated_reqs = []

    try:
        #  Gather current environment with pip freeze
        freeze_result = subprocess.run(
            ["pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )

        def is_version_on_pypi(pkg_name: str, version: str) -> bool:
            """
            Checks if the given package and exact version exist on PyPI via PyPI's JSON API.
            """
            try:
                url = f"https://pypi.org/pypi/{pkg_name}/json"
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    return False

                data = resp.json()
                if "releases" not in data:
                    return False

                return version in data["releases"]
            except Exception as e:
                print(f"Error verifying on PyPI: {pkg_name}=={version}: {e}")
                return False

        # Parse lines, ignoring editable/git references
        for line in freeze_result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Skip editable or local references
            if line.startswith("-e ") or "git+" in line or "@" in line:
                # print(f"Skipping non-PyPI requirement: {line}")
                continue

            # Extract package name and version when pinned with "=="
            if "==" in line:
                pkg_name, version = line.split("==", 1)
                pkg_name = pkg_name.strip()
                version = version.strip()

                # Check PyPI for the version
                if is_version_on_pypi(pkg_name, version):
                    validated_reqs.append(f"{pkg_name}=={version}")
                # else:
                # print(f"Warning: {pkg_name}=={version} not found on PyPI; skipping.")

        if not validated_reqs:
            raise RuntimeError(
                "No valid PyPI packages found to include in requirements."
            )

        # Step 4: Create a minimal requirements.in with validated pins
        with open(temp_req_in, "w") as f:
            for req in sorted(validated_reqs):
                f.write(req + "\n")

        # Step 5: Use pip-compile to finalize pinned dependencies
        compile_cmd = [
            "pip-compile",
            "--allow-unsafe",
            "--resolver=backtracking",
            temp_req_in,
            "--output-file",
            output_file,
        ]
        subprocess.run(
            compile_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        assert os.path.exists(output_file), (
            "Failed to resolve project requirements. Ensure all packages in your envoirnment are on PyPI."
        )

        return output_file

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate requirements: {e.stderr}") from e


def create_tp_job_script(tmp_dir: str, cmds: List[str]) -> str:
    """
    Create a tp-run.txt script for the project directory, saved to the tmp directory.
    """
    job_file = os.path.join(tmp_dir, "tp-run.txt")
    with open(job_file, "w") as f:
        for cmd in cmds:
            f.write(f"{cmd}\n")

    assert os.path.exists(job_file), f"Job script not created: {job_file}"

    return job_file


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

            req_file = os.path.join(tmp_dir, "tp-requirements.txt")
            job_file = os.path.join(tmp_dir, "tp-run.txt")

            if os.path.isfile(req_file):
                tar.add(req_file, arcname="tp-requirements.txt")
            else:
                Warning(f"{req_file} does not exist and will not be included.")

            if os.path.isfile(job_file):
                tar.add(job_file, arcname="tp-run.txt")
            else:
                Warning(f"{job_file} does not exist and will not be included.")

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


def dump_json_to_tp_toml(json: Dict) -> str:
    """
    Convert a JSON object to a TOML string and dumps to tp-config.toml in the cwd
    Arguments:
        json: The Dict to convert to TOML
    Returns:
        The filepath to the created TOML file
    """
    toml_str = toml.dumps(json)
    with open("tp-config.toml", "w+") as f:
        f.write(toml_str)

    tp_config_path = os.path.join(os.getcwd(), "tp-config.toml")

    if not os.path.exists(tp_config_path):
        raise FileNotFoundError("tp-config.toml was not created successfully")

    return tp_config_path


# TODO: remove this eventually
# This should just check if the proper keys exist and the types of the values, not the actual values
def soft_validate_config(config: Dict):
    """
    Validate the a job configuration
    """
    required_fields = {"commands", "optimization_priority"}
    if not all(field in config for field in required_fields):
        missing = required_fields - set(config.keys())
        raise ValueError(f"Missing required fields: {missing}")

    if not isinstance(config["commands"], list):
        raise ValueError("commands must be a list")
    if not all(isinstance(cmd, str) for cmd in config["commands"]):
        raise ValueError("all commands must be strings")

    if not isinstance(config["optimization_priority"], str):
        raise ValueError("optimization_priority must be a string")


def gen_job_metadata() -> Dict:
    """
    Generate metadata for a job
    """
    headers = {"Content-Type": "application/json"}
    payload = {"key": os.environ["TENSORPOOL_KEY"]}

    try:
        response = requests.post(
            f"{ENGINE}/job/gen", json=payload, headers=headers, timeout=10
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

    # Rougly check if under 900k tokens
    if count_tokens(query) > 9e6:
        raise RuntimeError("Project is too large to process")

    headers = {"Content-Type": "application/json"}
    payload = {
        "key": os.environ["TENSORPOOL_KEY"],
        "query": query,
        "dir_ctx": dir_ctx,
        "file_ctx": file_ctx,
    }
    # print("Payload:", payload)

    try:
        response = requests.post(
            f"{ENGINE}/translate",
            json=payload,
            headers=headers,
            timeout=30,  # limit is quite high...
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
    soft_validate_config(tp_config)

    merged = {**res, **tp_config}
    merged.pop("upload_url", None)  # Don't need the upload url at this point
    assert merged["is_valid_job"], "Job is not valid, we should not be here.."
    merged.pop("is_valid_job", None)

    headers = {"Content-Type": "application/json"}
    payload = {"key": os.environ["TENSORPOOL_KEY"], "config": merged}

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
