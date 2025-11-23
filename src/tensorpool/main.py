import os
import argparse
from tensorpool.helpers import (
    login,
    get_tensorpool_key,
    get_version,
    health_check,
    job_push,
    job_cancel,
    job_list,
    job_info,
    job_listen,
    job_pull,
    get_empty_tp_config,
    dump_file,
    download_files,
    cluster_create,
    cluster_destroy,
    cluster_list,
    cluster_info,
    cluster_edit,
    ssh_command,
    ssh_key_create,
    ssh_key_list,
    ssh_key_destroy,
    fetch_user_info,
    storage_create,
    storage_destroy,
    storage_attach,
    storage_detach,
    storage_list,
    storage_info,
    storage_edit,
    safe_input,
    safe_confirm,
    ENGINE,
)
from tensorpool.spinner import Spinner


def gen_tp_config(no_input: bool = False) -> None:
    """
    Command to generate a tp.[config].toml file
    """
    with Spinner(text="Fetching empty config..."):
        success, empty_config, message = get_empty_tp_config()

    if not success:
        print(f"Failed to fetch empty config: {message}")
        exit(1)

    if message:
        print(message)

    # Find a unique filename
    tp_config_path = "tp.config.toml"
    if os.path.exists(tp_config_path):
        count = 1
        while True:
            tp_config_path = f"tp.config{count}.toml"
            if not os.path.exists(tp_config_path):
                break
            count += 1

    print(f"Enter a name for the tp config, or press ENTER to use {tp_config_path}")
    new_name = safe_input("", default="", no_input=no_input)
    new_name = f"tp.{new_name}.toml" if new_name else None
    if new_name:
        tp_config_path = new_name

    save_success = dump_file(empty_config, tp_config_path)

    if not save_success:
        print("Failed to create new tp config")
        exit(1)

    print(f"{tp_config_path} created")
    print(f"Configure it to do `tp job push {tp_config_path}`")

    return


def main():
    parser = argparse.ArgumentParser(description="TensorPool https://tensorpool.dev")
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Disable interactive prompts (warning: may be destructive)",
    )

    subparsers = parser.add_subparsers(dest="command")

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Manage clusters",
    )

    cluster_subparsers = cluster_parser.add_subparsers(dest="cluster_command")

    cluster_create_parser = cluster_subparsers.add_parser(
        "create", help="Create a new cluster"
    )
    cluster_create_parser.add_argument(
        "-i",  # uh this is kinda weird but -i is standard for ssh private key / identity file,
        "--public-key",
        help="Path to your public SSH key (e.g. ~/.ssh/id_rsa.pub)",
        required=False,
    )
    cluster_create_parser.add_argument(
        "-t",
        "--instance-type",
        help="Instance type (e.g. 1xH100, 2xH100, 4xH100, 8xH100)",
        required=True,
    )
    cluster_create_parser.add_argument("--name", help="Cluster name (optional)")
    cluster_create_parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        help="Number of nodes (optional, required if instance type is 8xH100)",
    )
    cluster_create_parser.add_argument(
        "--deletion-protection",
        action="store_true",
        help="Enable deletion protection for the cluster (optional)",
    )
    cluster_create_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Skip confirmation prompts and create cluster immediately",
    )
    cluster_create_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the cluster to be fully provisioned before returning",
    )
    cluster_destroy_parser = cluster_subparsers.add_parser(
        "destroy", aliases=["rm"], help="Destroy a cluster"
    )
    cluster_destroy_parser.add_argument("cluster_id", help="Cluster ID to destroy")
    cluster_destroy_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Skip confirmation prompt and destroy cluster immediately",
    )
    cluster_destroy_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the cluster to be fully destroyed before returning",
    )
    list_parser = cluster_subparsers.add_parser(
        "list", aliases=["ls"], help="List available clusters"
    )
    list_parser.add_argument(
        "--org",
        "--organization",
        action="store_true",
        help="List all clusters in organization",
        dest="org",
    )

    info_parser = cluster_subparsers.add_parser(
        "info", help="Get detailed information about a cluster"
    )
    info_parser.add_argument("cluster_id", help="Cluster ID to get information about")

    cluster_edit_parser = cluster_subparsers.add_parser(
        "edit", help="Edit cluster properties"
    )
    cluster_edit_parser.add_argument("cluster_id", help="Cluster ID to edit")
    cluster_edit_parser.add_argument("--name", help="New name for the cluster")
    cluster_edit_parser.add_argument(
        "--deletion-protection",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Enable/disable deletion protection (true/false)",
    )

    # on_parser = cluster_subparsers.add_parser("on", help="Activate a cluster")
    # on_parser.add_argument("cluster_id", help="ID of the instance/cluster to turn on")

    # off_parser = cluster_subparsers.add_parser("off", help="Deactivate a cluster")
    # off_parser.add_argument("cluster_id", help="ID of the instance/cluster to turn off")

    storage_parser = subparsers.add_parser(
        "storage",
        aliases=["nfs"],
        help="Manage storage volumes",
    )

    storage_subparsers = storage_parser.add_subparsers(dest="storage_command")

    storage_create_parser = storage_subparsers.add_parser(
        "create", help="Create a new storage volume"
    )
    storage_create_parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=True,
        help="Size of the storage volume in GB",
    )
    storage_create_parser.add_argument("--name", help="Storage volume name (optional)")
    storage_create_parser.add_argument(
        "--deletion-protection",
        action="store_true",
        help="Enable deletion protection for the storage volume (optional)",
    )
    storage_create_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Skip confirmation prompts and create storage volume immediately",
    )

    storage_destroy_parser = storage_subparsers.add_parser(
        "destroy", aliases=["rm"], help="Destroy a storage volume"
    )
    storage_destroy_parser.add_argument("storage_id", help="Storage ID to destroy")
    storage_destroy_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Skip confirmation prompts and destroy storage volume immediately",
    )

    storage_list_parser = storage_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List all storage volumes",
    )
    storage_list_parser.add_argument(
        "--org",
        "--organization",
        action="store_true",
        help="List all storage volumes in organization",
        dest="org",
    )

    storage_info_parser = storage_subparsers.add_parser(
        "info", help="Get detailed information about a storage volume"
    )
    storage_info_parser.add_argument(
        "storage_id", help="Storage ID to get information about"
    )

    storage_edit_parser = storage_subparsers.add_parser(
        "edit", help="Edit storage volume properties"
    )
    storage_edit_parser.add_argument("storage_id", help="Storage ID to edit")
    storage_edit_parser.add_argument("--name", help="New name for the storage volume")
    storage_edit_parser.add_argument(
        "--deletion-protection",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        help="Enable/disable deletion protection (true/false)",
    )
    storage_edit_parser.add_argument(
        "-s",
        "--size",
        type=int,
        help="New size for the storage volume in GB (size can only be increased)",
    )

    storage_attach_parser = storage_subparsers.add_parser(
        "attach", help="Attach a storage volume to clusters"
    )
    storage_attach_parser.add_argument("storage_id", help="Storage ID to attach")
    storage_attach_parser.add_argument(
        "cluster_ids", nargs="+", help="Cluster IDs to attach the storage volume to"
    )
    storage_attach_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Skip confirmation prompts and attach storage volume immediately",
    )

    storage_detach_parser = storage_subparsers.add_parser(
        "detach", help="Detach a storage volume from clusters"
    )
    storage_detach_parser.add_argument("storage_id", help="Storage ID to detach")
    storage_detach_parser.add_argument(
        "cluster_ids", nargs="+", help="Cluster IDs to detach the volume from"
    )
    storage_detach_parser.add_argument(
        "--no-input",
        action="store_true",
        help="Skip confirmation prompts and detach storage volume immediately",
    )

    # Create job subparser for job-related commands
    job_parser = subparsers.add_parser("job", help="Manage jobs on TensorPool")
    job_subparsers = job_parser.add_subparsers(dest="job_command")

    job_subparsers.add_parser(
        "init",
        help="Create a new tp.config.toml file.",
    )

    run_parser = job_subparsers.add_parser("push", help="Run a job on TensorPool")
    run_parser.add_argument("tp_config_path", help="Path to a tp.{config}.toml file")
    # run_parser.add_argument(
    #     "--listen",
    #     action="store_true",
    #     help="Automatically listen to job output after pushing",
    # )

    job_list_parser = job_subparsers.add_parser(
        "list", aliases=["ls"], help="List jobs"
    )
    job_list_parser.add_argument(
        "--org",
        "--organization",
        action="store_true",
        help="List all jobs in organization",
        dest="org",
    )

    job_info_parser = job_subparsers.add_parser(
        "info", help="Get detailed information about a job"
    )
    job_info_parser.add_argument("job_id", help="Job ID to get information about")

    cancel_parser = job_subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")
    cancel_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the job to cancel before returning",
    )

    listen_parser = job_subparsers.add_parser("listen", help="Listen to a job")
    listen_parser.add_argument("job_id", help="ID of the job to listen to")

    pull_parser = job_subparsers.add_parser("pull", help="Pull job output files")
    pull_parser.add_argument("job_id", help="ID of the job to pull")
    pull_parser.add_argument(
        "-i",
        "--private-key",
        help="Path to SSH private key (e.g., ~/.ssh/id_ed25519)",
        required=False,
        default=None,
    )
    pull_parser.add_argument(
        "--path",
        nargs="*",
        help="Optional path(s) of specific files to pull (e.g., --path output/model.pt logs/training.log)",
    )
    pull_group = pull_parser.add_mutually_exclusive_group()
    pull_group.add_argument(
        "--force", action="store_true", help="Force overwrite existing files"
    )
    # pull_group.add_argument(
    #     "--dry-run",
    #     action="store_true",
    #     help="Preview files to be pulled without downloading",
    # )

    # job_subparsers.add_parser(
    #     "dashboard",
    #     aliases=[
    #         "dash",
    #         "jobs",
    #     ],
    #     help="Open the TensorPool dashboard",
    # )

    ssh_parser = subparsers.add_parser(
        "ssh", help="SSH into an instance"
    )
    ssh_parser.add_argument("instance_id", help="Instance ID to SSH into")
    ssh_parser.add_argument(
        "ssh_args",
        nargs=argparse.REMAINDER,
        help="Additional SSH arguments to pass through (e.g. -i, -o, -v)",
    )

    me_parser = subparsers.add_parser("me", help="Display user information and manage SSH keys")
    me_subparsers = me_parser.add_subparsers(dest="me_command")

    # Key management subcommand
    me_key_parser = me_subparsers.add_parser("sshkey", help="Manage SSH keys")
    me_key_subparsers = me_key_parser.add_subparsers(dest="sshkey_command")

    me_key_create_parser = me_key_subparsers.add_parser(
        "add", help="Add an SSH public key"
    )
    me_key_create_parser.add_argument("key_path", help="Path to SSH public key file")
    me_key_create_parser.add_argument("--name", help="Optional name for the SSH key")

    me_key_list_parser = me_key_subparsers.add_parser(
        "list", aliases=["ls"], help="List all SSH keys"
    )
    me_key_list_parser.add_argument(
        "--org",
        "--organization",
        action="store_true",
        help="List all SSH keys in organization",
        dest="org",
    )

    me_key_destroy_parser = me_key_subparsers.add_parser(
        "remove", aliases=["rm"], help="Remove an SSH key"
    )
    me_key_destroy_parser.add_argument("key_id", help="SSH key ID to remove")

    # Preferences subcommand
    me_preferences_parser = me_subparsers.add_parser(
        "preferences", help="Manage user preferences"
    )

    parser.add_argument("-v", "--version", action="version", version=f"{get_version()}")

    args = parser.parse_args()

    key = get_tensorpool_key()
    if not key:
        print("TENSORPOOL_KEY environment variable not found.")
        inp = safe_confirm(
            "Would you like to add it to .env? [Y/n] ",
            no_input=args.no_input,
            default="y",
        )
        if inp.lower() not in ["n", "no"]:
            if not login(no_input=args.no_input):
                print("Failed to set API key")
                exit(1)
        else:
            print("Please set TENSORPOOL_KEY environment variable before proceeding.")
            exit(1)
    else:
        os.environ["TENSORPOOL_KEY"] = key

    # Health check
    with Spinner(text="Authenticating..."):
        health_accepted, health_message = health_check()
    if not health_accepted:
        print(health_message)
        exit(1)
    else:
        if health_message:
            print(health_message)

    if args.command == "job":
        if args.job_command == "init":
            return gen_tp_config(no_input=args.no_input)
        elif args.job_command == "push":
            if not args.tp_config_path:
                print("Error: tp config path required")
                run_parser.print_help()
                return

            success, job_id = job_push(args.tp_config_path)
            if not success:
                exit(1)

            # # Auto-listen if --listen flag is set and we have a job_id
            # if args.listen and job_id:
            #     print(f"\nListening to job {job_id}...")
            #     listen_success, listen_message = job_listen(job_id)
            #     if listen_message:
            #         print(listen_message)
            #     if not listen_success:
            #         exit(1)

            return
        elif args.job_command == "listen":
            success, message = job_listen(args.job_id)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.job_command == "pull":
            if not args.job_id:
                print("Error: Job ID is required")
                pull_parser.print_help()
                return

            # Extract file paths if provided
            files = getattr(args, "path", None)
            if files is not None and len(files) == 0:
                files = None

            with Spinner(text="Fetching job files..."):
                download_map, msg = job_pull(
                    args.job_id,
                    files=files,
                    # dry_run=args.dry_run,
                    tensorpool_priv_key_path=args.private_key,
                )

            if not download_map:
                if msg:
                    print(msg)
                return

            num_files = len(download_map)
            if num_files == 0:
                print("No changed files to pull")
                return

            # if args.dry_run:
            #     print(f"Files that would be pulled ({num_files} total):")
            #     for file_path in download_map.keys():
            #         print(f"  {file_path}")
            #     return

            download_success = download_files(download_map, overwrite=args.force)

            if not download_success:
                print(
                    "Failed to download job files\nPlease try again or visit https://dashboard.tensorpool.dev/dashboard\nContact team@tensorpool.dev if this persists"
                )
                exit(1)

            print("Job files pulled successfully")
            return
        elif args.job_command == "cancel":
            success, message = job_cancel(args.job_id, no_input=args.no_input, wait=args.wait)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.job_command in ["list", "ls"]:
            success, message = job_list(org=args.org)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.job_command == "info":
            with Spinner(text="Fetching job info..."):
                success, message = job_info(args.job_id)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        else:
            job_parser.print_help()
            return
    if args.command == "cluster":
        if args.cluster_command == "create":
            identity_file_path = args.public_key if args.public_key else None

            success, final_message = cluster_create(
                identity_file_path,
                args.instance_type,
                args.name,
                args.num_nodes,
                args.deletion_protection,
                no_input=args.no_input,
                wait=args.wait,
            )
            if final_message:
                print(final_message)
            if not success:
                exit(1)
            return
        elif args.cluster_command in ["destroy", "rm"]:
            success, message = cluster_destroy(args.cluster_id, no_input=args.no_input, wait=args.wait)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.cluster_command in ["list", "ls"]:
            success, message = cluster_list(org=args.org)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.cluster_command == "info":
            with Spinner(text="Fetching cluster info..."):
                success, message = cluster_info(args.cluster_id)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.cluster_command == "edit":
            name = getattr(args, "name", None)
            deletion_protection = getattr(args, "deletion_protection", None)

            success, message = cluster_edit(
                args.cluster_id, name=name, deletion_protection=deletion_protection
            )
            if message:
                print(message)
            if not success:
                exit(1)
            return
        else:
            cluster_parser.print_help()
            return
    elif args.command == "ssh":
        instance_id = args.instance_id
        if not instance_id:
            print("Error: instance_id is required")
            ssh_parser.print_help()
            exit(1)

        ssh_args = (
            args.ssh_args if hasattr(args, "ssh_args") and args.ssh_args else []
        )
        success, message = ssh_command(instance_id, ssh_args)
        if message:
            print(message)
        if not success:
            exit(1)
        return
    elif args.command in ["storage", "nfs"]:
        if args.storage_command == "create":
            success, message = storage_create(
                args.name, args.size, args.deletion_protection, no_input=args.no_input
            )
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.storage_command in ["destroy", "rm"]:
            success, message = storage_destroy(args.storage_id, no_input=args.no_input)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.storage_command in ["list", "ls"]:
            with Spinner(text="Fetching storage volumes..."):
                success, message = storage_list(org=args.org)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.storage_command == "info":
            with Spinner(text="Fetching storage volume info..."):
                success, message = storage_info(args.storage_id)
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.storage_command == "edit":
            name = getattr(args, "name", None)
            deletion_protection = getattr(args, "deletion_protection", None)
            size = getattr(args, "size", None)
            with Spinner(text="Editing storage volume..."):
                success, message = storage_edit(
                    args.storage_id,
                    name=name,
                    deletion_protection=deletion_protection,
                    size=size,
                )
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.storage_command == "attach":
            success, message = storage_attach(
                args.storage_id, args.cluster_ids, no_input=args.no_input
            )
            if message:
                print(message)
            if not success:
                exit(1)
            return
        elif args.storage_command == "detach":
            success, message = storage_detach(
                args.storage_id, args.cluster_ids, no_input=args.no_input
            )
            if message:
                print(message)
            if not success:
                exit(1)
            return
        else:
            storage_parser.print_help()
            return
    elif args.command == "me":
        # Handle SSH key subcommands
        if args.me_command == "sshkey":
            if args.sshkey_command == "add":
                success, message = ssh_key_create(args.key_path, name=args.name)
                if message:
                    print(message)
                if not success:
                    exit(1)
                return
            elif args.sshkey_command in ["list", "ls"]:
                success, message = ssh_key_list(org=args.org)
                if message:
                    print(message)
                if not success:
                    exit(1)
                return
            elif args.sshkey_command in ["remove", "rm"]:
                success, message = ssh_key_destroy(args.key_id)
                if message:
                    print(message)
                if not success:
                    exit(1)
                return
            else:
                me_key_parser.print_help()
                return
        # Handle preferences subcommand
        elif args.me_command == "preferences":
            print("To manage your preferences, visit: https://tensorpool.dev/dashboard/profile")
            return
        # Handle default 'tp me' (display user info)
        else:
            with Spinner(text="Fetching user information..."):
                success, message = fetch_user_info()
            print(message)

            # Display engine URL if it's been overridden
            if os.environ.get("TENSORPOOL_ENGINE"):
                print(f"\nTENSORPOOL_ENGINE={ENGINE}")

            if not success:
                exit(1)
            return

    parser.print_help()
    return

    # text = " ".join(args.query)
    # print(f"You said: {text}")


if __name__ == "__main__":
    main()
