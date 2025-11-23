<p align="center">
  <img src="https://tensorpool.dev/images/tp-full-dark-gh-repo.png" alt="TensorPool Logo" width="600">
</p>

# TensorPool CLI

[![PyPI version](https://badge.fury.io/py/tensorpool.svg)](https://badge.fury.io/py/tensorpool)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-tensorpool.dev-blue)](https://docs.tensorpool.dev/)
[![Slack Community](https://img.shields.io/badge/slack-join-brightgreen.svg)](https://tensorpool.dev/slack)

The TensorPool CLI is a git-style interface for running ML training jobs and accessing on-demand multi-node GPU clusters.


## Quick Start

```bash
# Install
pip install tensorpool
```

**Get started in minutes:**
- [Jobs Quick Start](https://docs.tensorpool.dev/quickstart) - Submit your first job
- [Clusters Quick Start](https://docs.tensorpool.dev/clusters-quickstart) - Create your first cluster

## Usage

```bash
% tp --help
usage: tp [-h] [--no-input] [-v] {cluster,storage,nfs,job,ssh,me} ...

TensorPool https://tensorpool.dev

positional arguments:
  {cluster,storage,nfs,job,ssh,me}
    cluster             Manage clusters
    storage             Manage storage volumes
    job                 Manage jobs on TensorPool
    ssh                 SSH into an instance
    me                  Display user information and manage SSH keys

options:
  -h, --help            show this help message and exit
  --no-input            Disable interactive prompts
  -v, --version         show program's version number and exit
```

**Learn more about the CLI at [docs.tensorpool.dev/cli/overview](https://docs.tensorpool.dev/cli/overview)**

## Documentation

**Full documentation at [docs.tensorpool.dev](https://docs.tensorpool.dev/)**

- [Installation Guide](https://docs.tensorpool.dev/installation)
- [CLI Reference](https://docs.tensorpool.dev/cli/overview)
- [API Reference](https://docs.tensorpool.dev/api/introduction)
- [Instance Types](https://docs.tensorpool.dev/resources/instance-types)

## Support

- [Documentation](https://docs.tensorpool.dev/)
- [Slack Community](https://tensorpool.dev/slack)
- [Discord](https://discord.gg/9cSxZ7dxSk)
- [team@tensorpool.dev](mailto:team@tensorpool.dev)
- [Twitter/X](https://x.com/TensorPool)
- [Website](https://tensorpool.dev)
