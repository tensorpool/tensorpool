import os
import argparse
from tensorpool.helpers import (
    login,
    get_tensorpool_key,
    get_version,
    health_check,
    load_tp_config,
    get_project_files,
    upload_files,
    job_init,
    job_submit,
    save_empty_tp_config,
    autogen_tp_config,
    construct_proj_ctx,
    get_proj_paths,
    listen_to_job,
    dump_tp_toml,
    fetch_dashboard,
    job_pull,
    download_files,
    job_cancel,
    get_unique_config_path,
)
from tensorpool.spinner import Spinner
from typing import Optional, List


def gen_tp_config(prompt):
    """
    Command to generate a tp.[config].toml file
    """

    if not prompt:
        print(
            "To create a new tp config, run `tp config new` or `tp config [NL prompt]` to auto-generate it."
        )
        return
    elif prompt.lower() == "new":
        tp_config_path = get_unique_config_path()

        # Ask the user if they want this name, or if they want to specify a different name
        print(f"Enter a name for the tp config, or press ENTER to use {tp_config_path}")
        new_name = input()
        new_name = f"tp.{new_name}.toml" if new_name else None
        print(new_name)
        if new_name:
            tp_config_path = new_name

        save_success = save_empty_tp_config(tp_config_path)

        if not save_success:
            print("Failed to create new tp config")
            return

        print(f"New tp config created: {tp_config_path}")
        print(f"Configure it and then do `tp run {tp_config_path}` to execute the job")

        return

    with Spinner(text="Indexing project..."):
        file_paths, filtered_file_contents = construct_proj_ctx(get_proj_paths())

    with Spinner(text="Configuring..."):
        translation_success, tp_config, message = autogen_tp_config(
            prompt, file_paths, filtered_file_contents
        )

    if not translation_success:
        if message is None:
            raise RuntimeError(
                "A new tp.config.toml failed to autogenerate\nPlease try again or run `tp config new` to manually configure it."
            )
        else:
            print(message)
            return

    with Spinner():
        tp_config_path = get_unique_config_path()

    # Ask the user if they want this name, or if they want to specify a different name
    print(f"Enter a name for the tp config, or press ENTER to use {tp_config_path}")
    new_name = input()
    new_name = f"tp.{new_name}.toml" if new_name else None
    if new_name:
        tp_config_path = new_name

    if os.path.exists(tp_config_path):
        print(f"File {tp_config_path} already exists. Overwrite? [Y/n]", end=" ")
        response = input()
        if response.lower() in ["n", "no"]:
            print("Exiting...")
            return

    with Spinner(text="Saving tp config..."):
        tp_config_save_success = dump_tp_toml(tp_config, tp_config_path)

    if not tp_config_save_success:
        print(f"Failed to save tp config to {tp_config_path}")
        return

    if message:
        print(message)

    return


def run(
    tp_config_path,
    use_cache: Optional[str] = None,
    detach: bool = False,
    pull_on_complete: bool = False,
):
    """
    Run a job
    use_cache is a job ID to use as a cache, if empty the last job will be used
    """
    assert os.path.exists(tp_config_path), f"{tp_config_path} not found"

    with Spinner(text="Indexing project..."):
        tp_config = load_tp_config(tp_config_path)
        project_files = get_project_files(tp_config.get("ignore"))
        num_files = len(project_files)

    if num_files == 0:
        print("No files found in project. Are you in the right directory?")
        return
    elif num_files >= 1000:
        # TODO: mention ignore and point to docs
        print(f"You have {num_files} files in your project. This may take a while.")

    with Spinner(text="Initializing job..."):
        # This can take a while for big projects...
        message, job_id, upload_map = job_init(tp_config, project_files, use_cache)

    if message:
        print(message)

    # Note: there may be no new files to upload (upload_map = {})
    upload_success = upload_files(upload_map)

    if not upload_success:
        print("Project upload failed. Please try again.")
        return

    with Spinner(text="Submitting job..."):
        job_submit_success, post_submit_message = job_submit(job_id, tp_config)

    print(post_submit_message)
    if job_submit_success and not detach:
        listen(job_id, pull_on_complete=pull_on_complete)

    return


def listen(
    job_id: str, pull_on_complete: bool = False, overwrite_on_pull: bool = False
):
    if not job_id:
        print("Error: Job ID required")
        print("Usage: tp listen <job_id>")
        return

    completed = listen_to_job(job_id)

    if completed and pull_on_complete:
        pull(job_id, files=None, overwrite=overwrite_on_pull)

    return


def pull(
    job_id: str,
    files: Optional[List[str]] = None,
    overwrite: bool = False,
    preview: bool = False,
):
    # if not job_id:
    #     print("Error: Job ID required")
    #     print("Usage: tp pull <job_id>")
    #     return
    if files and len(files) > 100:
        print(f"{len(files)} files requested, this may take a while")

    with Spinner(text="Pulling job..."):
        download_map, msg = job_pull(job_id, files, preview)

    if not download_map:
        if msg:
            print(msg)
        return

    num_files = len(download_map)
    if num_files == 0:
        print("No changed files to pull")
        return

    download_success = download_files(download_map, overwrite)

    if not download_success:
        print(
            "Failed to download job files\nPlease try again or visit https://dashboard.tensorpool.dev/dashboard\nContact team@tensorpool.dev if this persists"
        )
        return

    print("Job files pulled successfully")

    return


def cancel(job_id: str):
    cancel_success, message = job_cancel(job_id)
    print(message)


def dashboard():
    dash = fetch_dashboard()

    if dash:
        print(dash)
    else:
        print("Failed fetch dashboard, visit https://tensorpool.dev")

    return


def main():
    parser = argparse.ArgumentParser(
        description="TensorPool is the easiest way to use cloud GPUs. https://tensorpool.dev"
    )

    subparsers = parser.add_subparsers(dest="command")
    gen_parser = subparsers.add_parser(
        "config",
        help="Generate a tp.config.toml job configuration file",
        description="""
Generate a TensorPool configuration file (tp.config.toml) in two ways:
1. Create an empty template: tp config new
2. Auto-generate from natural language: tp config "train on 4 GPUs for 3 hours"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    gen_parser.add_argument(
        "config",
        nargs="*",
        metavar="PROMPT",
        help='Either "new" to create an empty config, or a natural language description of your job'
    )

    run_parser = subparsers.add_parser("run", help="Run a job on TensorPool")
    run_parser.add_argument(
        "tp_config_path", nargs="?", help="Path to a tp.[config].toml file"
    )
    run_parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Don't check your job cache for previously uploaded files",
    )
    run_parser.add_argument(
        "--detach", action="store_true", help="Run the job in the background"
    )
    # TODO: listen & docuement these
    # run_parser.add_argument("--gpu", help="GPU type to use")
    # run_parser.add_argument("--gpu-count", help="Number of GPUs to use")
    # run_parser.add_argument("--vcpus", help="Number of vCPUs to use")
    # run_parser.add_argument("--memory", help="Amount of memory to use in GB")

    listen_parser = subparsers.add_parser("listen", help="Listen to a job")
    listen_parser.add_argument("job_id", help="ID of the job to listen to")
    listen_parser.add_argument(
        "--pull", action="store_true", help="Pull the job files after listening"
    )
    listen_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files if pulling"
    )
    pull_parser = subparsers.add_parser("pull", help="Pull a job")
    pull_parser.add_argument("job_id", nargs="?", help="ID of the job to pull")
    pull_parser.add_argument("files", nargs="*", help="List of filenames to pull")
    pull_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    pull_parser.add_argument(
        "--preview", action="store_true", help="Preview the files to be pulled"
    )

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_ids", nargs="+", help="IDs of the job(s) to cancel")

    # cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        aliases=[
            "dash",
            "jobs",
        ],
        help="Open the TensorPool dashboard",
    )

    parser.add_argument("-v", "--version", action="version", version=f"{get_version()}")

    args = parser.parse_args()

    key = get_tensorpool_key()
    if not key:
        print("TENSORPOOL_KEY environment variable not found.")
        inp = input("Would you like to add it to .env? [Y/n] ")
        if inp.lower() not in ["n", "no"]:
            if not login():
                print("Failed to set API key")
                return
        else:
            print("Please set TENSORPOOL_KEY environment variable before proceeding.")
            return
    else:
        os.environ["TENSORPOOL_KEY"] = key

    # Health check
    with Spinner(text="Authenticating..."):
        health_accepted, health_message = health_check()
    if not health_accepted:
        print(health_message)
        return
    else:
        if health_message:
            print(health_message)

    if args.command == "config":
        prompt = " ".join(args.config)
        return gen_tp_config(prompt)
    elif args.command == "run":
        if not args.tp_config_path:
            # Find all tp config files in current directory
            config_files = [
                f
                for f in os.listdir(".")
                if f.startswith("tp.") and f.endswith(".toml")
            ]

            if not config_files:
                print("No tp config files found. Create one with 'tp config new'")
                return
            elif len(config_files) == 1:
                args.tp_config_path = config_files[0]
            else:
                print(
                    "Select config file(s) to run (comma-separated numbers, e.g. 3, or 1,2,3):"
                )
                for idx, file in enumerate(config_files):
                    print(f"{idx + 1}. {file}")

                while True:
                    try:
                        selections = input("Select config(s) to use: ").split(",")
                        selections = [int(s.strip()) - 1 for s in selections]
                        if all(0 <= s < len(config_files) for s in selections):
                            for config_file in [config_files[s] for s in selections]:
                                run(
                                    config_file,
                                    args.skip_cache,
                                    True
                                    if len(selections) > 1
                                    else args.detach,  # force detach for multiple jobs
                                    pull_on_complete=False,
                                )

                            return
                        print("Invalid selection(s), try again")
                    except ValueError:
                        print("Please enter comma-separated numbers")

        return run(
            args.tp_config_path, args.skip_cache, args.detach, pull_on_complete=False
        )
    elif args.command == "listen":
        return listen(args.job_id, args.pull, args.overwrite)
    elif args.command == "pull":
        # Check if job_id is a snowflake
        return pull(args.job_id, args.files, args.overwrite, args.preview)
    elif args.command == "cancel":
        for job_id in args.job_ids:
            cancel(job_id)
        return
    #
    elif (
        args.command == "dashboard" or args.command == "dash" or args.command == "jobs"
    ):
        return dashboard()

    parser.print_help()
    return

    # text = " ".join(args.query)
    # print(f"You said: {text}")


if __name__ == "__main__":
    main()
