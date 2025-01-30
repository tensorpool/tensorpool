import os
import argparse
from dataclasses import dataclass
import datetime
from tensorpool.helpers import (
    login,
    get_tensorpool_key,
    get_version,
    health_check,
    get_proj_paths,
    construct_proj_ctx,
    is_utf8_encoded,
    get_file_contents,
    create_proj_tarball,
    gen_job_metadata,
    upload_with_progress,
    dump_tp_toml,
    translate_job,
    submit_job,
)
from tensorpool.spinner import Spinner
import tempfile
import toml


def main():
    parser = argparse.ArgumentParser(
        description="TensorPool is the easiest way to execute ML jobs in the cloud. https://tensorpool.dev"
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{get_version()}",
        help="show version",
    )

    parser.add_argument(
        "query",
        nargs="*",
        help='Command (jobs/dashboard) or natural language query (e.g. "run main.py on a T4")',
    )

    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    first_arg = args.query[0].lower()
    if first_arg in ("dashboard", "jobs", "dash"):
        print("https://tensorpool.dev/dashboard")
        return
    elif first_arg in ("version"):
        parser.parse_args(["--version"])
        return

    text = " ".join(args.query)
    # print(f"You said: {text}")

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

    with Spinner(text="Authenticating..."):
        health_accepted, health_message = health_check()
    if not health_accepted:
        print(health_message)
        return
    else:
        if health_message:
            print(health_message)

    with Spinner(text="Indexing project..."):
        file_paths, filtered_file_contents = construct_proj_ctx(get_proj_paths())

    use_existing = False
    if os.path.exists(os.path.join(os.getcwd(), "tp-config.toml")):
        print("Found existing tp-config.toml, would you like to use it? [Y/n]", end=" ")
        response = input()
        if response.lower() not in ["n", "no"]:
            use_existing = True
            try:
                config = toml.load("tp-config.toml")
            except Exception as e:
                print(f"Error loading tp-config.toml: {str(e)}")
                return

    with Spinner(text="Configuring..."):
        if use_existing:
            res = gen_job_metadata()
            # is_valid_job always comes back as True
            res.update(config)
        else:
            translated = translate_job(
                query=text, dir_ctx=file_paths, file_ctx=filtered_file_contents
            )
            job_metadata = gen_job_metadata()
            res = {
                **job_metadata,
                **translated,
            }  # overwrite job_metadata with translated

    if res["is_valid_job"]:
        filtered_res = {
            k: v
            for k, v in res.items()
            if k not in ["is_valid_job", "upload_url", "id", "refusal"]
        }
        tp_config_path = dump_tp_toml(filtered_res)

        print(f"Configuration saved to {os.path.relpath(tp_config_path)}")
        print("Please confirm and modify it if needed.")
        print("Press ENTER to submit for execution.")
        input()

        if not os.path.exists(tp_config_path):
            raise FileNotFoundError(f"{tp_config_path} disappeared!")

        try:
            user_updated_config: dict = toml.load(tp_config_path)
        except toml.TomlDecodeError as e:
            print(f"tp-config.toml invalid\nError: {str(e)}")
            return

        # package and send the project
        with tempfile.TemporaryDirectory() as tmp_dir:

            # with Spinner(text="Resolving dependencies..."):
            #     tp_reqs_path = create_tp_reqs(tmp_dir)
            #     tp_job_path = create_tp_job_script(
            #         tmp_dir, user_updated_config["commands"]
            #     )
            with Spinner(text="Packaging project..."):
                # tp_job_path = create_tp_job_script(tmp_dir, res["commands"])
                tp_tarball_path = create_proj_tarball(tmp_dir)

            upload_successful = upload_with_progress(tp_tarball_path, res["upload_url"])
        if not upload_successful:
            print("Project upload failed. Please try again.")
            return

        with Spinner(text="Submitting job..."):
            # TODO: update fn signature, no need for res
            job_link = submit_job(
                res, user_updated_config
            )
        if job_link is not None:
            print(f"Job {res['id']} submitted successfully.")
            print(f"See its status at {job_link}")

    else:
        print(res["refusal"])


if __name__ == "__main__":
    main()
