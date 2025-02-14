import os
import argparse
from tensorpool.helpers import (
    login,
    get_tensorpool_key,
    get_version,
    health_check,
    load_tp_config,
    snapshot_proj_state,
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
)
from tensorpool.spinner import Spinner


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
        tp_config_path = "tp.config.toml"
        # Find a unique default name
        if os.path.exists(tp_config_path):
            count = 1
            while True:
                tp_config_path = f"tp.config{count}.toml"
                if not os.path.exists(tp_config_path):
                    break
                count += 1

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
        tp_config_path = "tp.config.toml"
        # Find a unique default name
        if os.path.exists(tp_config_path):
            count = 1
            while True:
                tp_config_path = f"tp.config{count}.toml"
                if not os.path.exists(tp_config_path):
                    break
                count += 1

    # Ask the user if they want this name, or if they want to specify a different name
    print(
        f"Enter desired name for the tp config, or press ENTER to use {tp_config_path}"
    )
    new_name = input()
    new_name = f"tp.{new_name}.toml" if new_name else None
    if new_name:
        tp_config_path = new_name

    if os.path.exists(tp_config_path):
        print(f"File {tp_config_path} already exists. Overwrite? [Y/n]", end=" ")
        response = input()
        if response.lower() not in ["n", "no"]:
            print("Exiting...")
            return

    with Spinner(text="Saving tp config..."):
        tp_config_save_success = dump_tp_toml(tp_config, tp_config_path)

    # TODO

    if message:
        print(message)

    return


def run(tp_config_path, detach=False):
    assert os.path.exists(tp_config_path), f"{tp_config_path} not found"

    with Spinner(text="Indexing project..."):
        tp_config = load_tp_config(tp_config_path)
        project_state_snapshot = snapshot_proj_state()

    with Spinner(text="Initializing job..."):
        job_init_res = job_init(tp_config, project_state_snapshot)
        status = job_init_res.get("status", None)
        message = job_init_res.get("message", None)

    if message:
        print(message)
    if status != "success":
        return

    job_id = job_init_res.get("id", None)
    if not job_id:
        print("Error: No job ID recieved. Please contact team@tensorpool.dev")
        return

    upload_map = job_init_res.get("upload_map", None)
    # Note: there may be no new files to upload
    upload_success = upload_files(upload_map)

    if not upload_success:
        print("Project upload failed. Please try again.")
        return

    with Spinner(text="Submitting job..."):
        job_submit_success, post_submit_message = job_submit(job_id, tp_config)

    print(post_submit_message)
    if job_submit_success and not detach:
        listen_to_job(job_id)

    return


def listen(job_id):
    # TODO: use a snapshot to infer the job_id
    if not job_id:
        print("Error: Job ID required")
        print("Usage: tp listen <job_id>")
        return

    listen_to_job(job_id)

    return


def pull(job_id):
    # if not job_id:
    #     print("Error: Job ID required")
    #     print("Usage: tp pull <job_id>")
    #     return

    with Spinner(text="Pulling job..."):
        snapshot = None
        if job_id is None:
            snapshot = snapshot_proj_state()
        download_map, msg = job_pull(job_id, snapshot)

    if not download_map:
        if msg:
            print(msg)
        return

    download_success = download_files(download_map)

    if not download_success:
        print(
            "Failed to download job files\nPlease try again or visit https://dashboard.tensorpool.dev/dashboard\nContact team@tensorpool.dev if this persists"
        )
        return

    print("Job files pulled successfully")

    return


def cancel():
    pass


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
        "config", help="generate a tp-config.toml job configuration"
    )
    gen_parser.add_argument(
        "config", nargs="*", help="Configuration name or natural language prompt"
    )
    # TODO: improve how this shows in --help?

    run_parser = subparsers.add_parser("run", help="Run a job on TensorPool")
    run_parser.add_argument(
        "tp_config_path", nargs="?", help="Path to a tp.[config].toml file"
    )
    run_parser.add_argument(
        "--detach", action="store_true", help="Run the job in the background"
    )

    listen_parser = subparsers.add_parser("listen", help="Listen to a job")
    listen_parser.add_argument("job_id", help="ID of the job to listen to")

    pull_parser = subparsers.add_parser("pull", help="Pull a job")
    pull_parser.add_argument("job_id", nargs="?", help="ID of the job to pull")
    # cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        # TODO: these don't work?
        # aliases=[
        #     "dash",
        #     "jobs",
        # ],
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
                print("Select a config file to run:")
                for idx, file in enumerate(config_files):
                    print(f"{idx + 1}. {file}")

                while True:
                    try:
                        selection = int(input("Select a config to use: ")) - 1
                        if 0 <= selection < len(config_files):
                            args.tp_config_path = config_files[selection]
                            break
                        print("Invalid selection, try again")
                    except ValueError:
                        print("Please enter a number")

        return run(args.tp_config_path, args.detach)
    elif args.command == "listen":
        return listen(args.job_id)
    elif args.command == "pull":
        return pull(args.job_id)
    # elif args.command == 'cancel':
    #     return cancel()
    #
    elif args.command == "dashboard":
        return dashboard()

    parser.print_help()
    return

    # text = " ".join(args.query)
    # print(f"You said: {text}")


if __name__ == "__main__":
    main()
