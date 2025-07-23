# TensorPool

TensorPool is the easiest way to deploy and manage GPU clusters, at a fraction of the cost of traditional cloud providers.

## Features
- **Zero Cloud Setup**: No GCP, no AWS, no Docker, no cloud accounts required
- **Instant GPU Clusters**: Deploy multi-node GPU clusters with a single command
- **Flexible Storage**: Attach and detach NFS volumes across your clusters
- **>50% cheaper than traditional cloud providers**: TensorPool aggregates demand across multiple cloud provides and thus offers GPUs at a fraction of market price. Check out our pricing here: [TensorPool Pricing](https://tensorpool.dev/pricing)
- **High-Performance Networking**: All clusters come with high-speed interconnects for distributed training

## Prerequisites
1. Create an account at [tensorpool.dev](https://tensorpool.dev)
2. Get your API key from the [dashboard](https://dashboard.tensorpool.dev/dashboard)
3. Install the CLI:
```bash
pip install tensorpool
```
4. Generate SSH keys (if you don't have them):
```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa
```

## Quick Start

### 1. Create Your First GPU Cluster
Deploy a single H100 node cluster:
```bash
tp cluster create -i ~/.ssh/id_rsa.pub -t 1xH100 --name my-training-cluster
```

For multi-node training, create a 4-node H100 cluster:
```bash
tp cluster create -i ~/.ssh/id_rsa.pub -t 8xH100 -n 4 --name distributed-training
```

### 2. List Your Clusters
```bash
tp cluster list
```

### 3. SSH Into Your Cluster
Once your cluster is ready, you'll receive the connection details. SSH into your nodes and start training!

### 4. Clean Up
When you're done, destroy your cluster:
```bash
tp cluster destroy <cluster_id>
```

## Core Commands

### Cluster Management
- `tp cluster create` - Deploy a new GPU cluster
- `tp cluster list` - View all your clusters
- `tp cluster info <cluster_id>` - Get detailed information about a cluster
- `tp cluster destroy <cluster_id>` - Terminate a cluster

### Network File System (NFS)
- `tp nfs create` - Create a new NFS volume
- `tp nfs list` - View all your NFS volumes
- `tp nfs attach <storage_id> <cluster_ids>` - Attach storage to one or more clusters
- `tp nfs detach <storage_id> <cluster_ids>` - Detach storage from one or more clusters
- `tp nfs destroy <storage_id>` - Delete an NFS volume

### Account Management
- `tp me` - View your account information and usage

## Supported Instance Types

| Instance Type | GPUs | GPU Model |
|---------------|------|-----------|
| `1xH100` | 1 | H100 |
| `2xH100` | 2 | H100 |
| `4xH100` | 4 | H100 |
| `8xH100` | 8 | H100 |

*More instance types coming soon!*

## Command Reference

### Creating Clusters

```bash
tp cluster create -i <public_key_path> -t <instance_type> [options]
```

**Required Arguments:**
- `-i, --public-key`: Path to your public SSH key (e.g., `~/.ssh/id_rsa.pub`)
- `-t, --instance-type`: Instance type (`1xH100`, `2xH100`, `4xH100`, `8xH100`)

**Optional Arguments:**
- `--name`: Custom cluster name
- `-n, --num-nodes`: Number of nodes (required for `8xH100` instance type)

**Examples:**
```bash
# Single node H100
tp cluster create -i ~/.ssh/id_rsa.pub -t 1xH100 --name dev-cluster

# 2-node cluster with 8xH100 each (16 GPUs total)
tp cluster create -i ~/.ssh/id_rsa.pub -t 8xH100 -n 2 --name large-training
```

### Listing Clusters

```bash
tp cluster list [--org]
```

**Optional Arguments:**
- `--org, --organization`: List all clusters in your organization

### Destroying Clusters

```bash
tp cluster destroy <cluster_id>
```

**Arguments:**
- `cluster_id`: The ID of the cluster to destroy

## NFS Storage Command Reference

### Creating NFS Volumes

```bash
tp nfs create -s <size_in_gb> [--name <name>]
```

**Required Arguments:**
- `-s, --size`: Size of the NFS volume in GB

**Optional Arguments:**
- `--name`: Custom volume name

**Examples:**
```bash
# Create a 500GB volume
tp nfs create -s 500 --name training-data

# Create a 1TB volume with auto-generated name
tp nfs create -s 1000
```

### Listing NFS Volumes

```bash
tp nfs list [--org]
```

**Optional Arguments:**
- `--org, --organization`: List all NFS volumes in your organization

### Attaching NFS Volumes

```bash
tp nfs attach <storage_id> <cluster_ids> [<cluster_ids> ...]
```

**Arguments:**
- `storage_id`: The ID of the storage volume to attach
- `cluster_ids`: One or more cluster IDs to attach the volume to

### Detaching NFS Volumes

```bash
tp nfs detach <storage_id> <cluster_ids> [<cluster_ids> ...]
```

**Arguments:**
- `storage_id`: The ID of the storage volume to detach
- `cluster_ids`: One or more cluster IDs to detach the volume from

### Destroying NFS Volumes

```bash
tp nfs destroy <storage_id>
```

**Arguments:**
- `storage_id`: The ID of the storage volume to destroy

### Creating NFS Volumes
```bash
tp nfs create -s 1000 --name shared-datasets
```

### Attaching Storage to Clusters
```bash
# Attach to a single cluster
tp nfs attach <storage_id> <cluster_id>

# Attach to multiple clusters
tp nfs attach <storage_id> <cluster_id_1> <cluster_id_2> <cluster_id_3>
```

### Detaching Storage from Clusters
```bash
# Detach from a single cluster
tp nfs detach <storage_id> <cluster_id>

# Detach from multiple clusters
tp nfs detach <storage_id> <cluster_id_1> <cluster_id_2>
```

The NFS volume will be automatically mounted on all nodes in the cluster, at `/mnt/data/`.

## Best Practices

- **SSH Key Management**: Always use strong SSH keys and keep your private keys secure
- **Cluster Naming**: Use descriptive names for your clusters to easily identify them
- **Cost Management**: Destroy clusters when not in use to avoid unnecessary charges
- **Data Persistence**: Use NFS volumes for important data that needs to persist across cluster lifecycles
- **Multi-Node Training**: For distributed training, ensure your training scripts are configured for multi-node setups
- **Monitoring**: Regularly check `tp cluster list` to monitor your active resources

## Getting Help

- **Documentation**: [tensorpool.dev](https://tensorpool.dev)
- **Community**: [Join our Discord](https://discord.gg/Kzan7CZauT)
- **Support**: team@tensorpool.dev
- **Updates**: Follow us on [Twitter/X](https://x.com/TensorPool)

## Why TensorPool?

- **Simplicity**: Deploy GPU clusters without the complexity of cloud setup, networking, or quota management
- **Flexibility**: Scale from single GPUs to massive multi-node clusters instantly
- **Cost Effective**: Aggregated GPU capacity from multiple providers means better pricing
- **Performance**: High-speed networking and optimized configurations for ML workloads
- **No Lock-in**: Standard SSH access means you can use any tools and frameworks you prefer

Ready to scale your ML training? Get started at [tensorpool.dev](https://tensorpool.dev)!