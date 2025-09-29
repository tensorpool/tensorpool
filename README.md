# TensorPool

TensorPool is the easiest way to deploy and manage GPU clusters, at a fraction of the cost of traditional cloud providers.

## Features
- **Instant GPU Clusters**: Deploy multi-node GPU clusters with a single command
- **Flexible Storage**: Attach and detach NFS volumes across your clusters
- **High-Performance Networking**: All clusters come with high-speed interconnects for distributed training

## Prerequisites
1. Create an account at [tensorpool.dev](https://tensorpool.dev)
2. Get your API key from the [dashboard](https://dashboard.tensorpool.dev/api-key)
3. Set your API key as an environment variable:
```bash
export TENSORPOOL_API_KEY="your_api_key_here"
```
   Or add it to your shell profile for persistence:

   **For bash users:**
   ```bash
   echo 'export TENSORPOOL_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc
   ```

   **For zsh users:**
   ```bash
   echo 'export TENSORPOOL_API_KEY="your_api_key_here"' >> ~/.zshrc
   source ~/.zshrc
   ```
4. Install the CLI:
```bash
pip install tensorpool
```
5. Generate SSH keys (if you don't have them):
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

**Example SSH command**
```bash
ssh tensorpool@192.168.1.42
```

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
- `tp nfs create -s <size_gb>` - Create a new NFS volume
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
| `1xH200` | 1 | H200 |
| `2xH200` | 2 | H200 |
| `4xH200` | 4 | H200 |
| `8xH200` | 8 | H200 |
| `1xB200` | 1 | B200 |
| `2xB200` | 2 | B200 |
| `4xB200` | 4 | B200 |
| `8xB200` | 8 | B200 |
| `1xMI300X` | 1 | MI300X |

*More instance types coming soon!*

## Command Reference

### Creating Clusters

```bash
tp cluster create -i <public_key_path> -t <instance_type> [options]
```

**Required Arguments:**
- `-i, --public-key`: Path to your public SSH key (e.g., `~/.ssh/id_rsa.pub`)
- `-t, --instance-type`: Instance type (`1xH100`, `2xH100`, `4xH100`, `8xH100`, `1xH200`, `2xH200`, `4xH200`, `8xH200`, `1xB200`, `2xB200`, `4xB200`, `8xB200`, `1xMI300X`)

**Optional Arguments:**
- `--name`: Custom cluster name
- `-n, --num-nodes`: Number of nodes (only supported for `8xH100` instance type for multi-node)

**Examples:**
```bash
# Single H100
tp cluster create -i ~/.ssh/id_rsa.pub -t 1xH100

# Single node H200
tp cluster create -i ~/.ssh/id_rsa.pub -t 8xH200

# Single node B200
tp cluster create -i ~/.ssh/id_rsa.pub -t 8xB200

# Single node MI300X
tp cluster create -i ~/.ssh/id_rsa.pub -t 8xMI300X

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

**Note:** NFS volumes can only be attached to multi-node clusters (clusters with 2 or more nodes).

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

### NFS Storage Example Workflow

```bash
# 1. Create a 1TB NFS volume named "shared-datasets"
tp nfs create -s 1000 --name shared-datasets

# 2. Attach the volume to a single cluster
tp nfs attach <storage_id> <cluster_id>

# 3. Attach the volume to multiple clusters
tp nfs attach <storage_id> <cluster_id_1> <cluster_id_2> <cluster_id_3>

# 4. Detach the volume from a single cluster
tp nfs detach <storage_id> <cluster_id>

# 5. Detach the volume from multiple clusters
tp nfs detach <storage_id> <cluster_id_1> <cluster_id_2>
```

You replace `<storage_id>` and `<cluster_id>` with your actual IDs as needed.

## Storage Locations (Multi-node Clusters)

### Local NVMe Storage
Each cluster node comes with high-performance local NVMe storage mounted at:
```
/mnt/local
```
### NFS Volume Mount Points
When you attach an NFS volume to your cluster, it will be mounted at:
```
/mnt/nfs
```

**Convenient Symlinks:**
For easy access, the storage locations are also symlinked in your home directory:
- Local storage: `~/local` → `/mnt/local`
- NFS storage: `~/nfs` → `/mnt/nfs`

## Best Practices

- **SSH Key Management**: Keep your private ssh keys secure
- **Cluster Naming**: Use descriptive names for your clusters to easily identify them
- **Cost Management**: Destroy clusters when not in use to avoid unnecessary charges
- **Data Persistence**: Use NFS volumes for important data that needs to persist across cluster lifecycles
- **Multi-Node Training**: For distributed training, ensure your training scripts are configured for multi-node setups
- **Monitoring**: Regularly check `tp cluster list` to monitor your active resources

## Getting Help

- **Documentation**: [tensorpool.dev](https://tensorpool.dev)
- **Community**: [Join our Slack](https://join.slack.com/t/tensorpoolpublic/shared_invite/zt-3aw1a1ncw-vF9vTjmqiGbOlhhcnzA03w)
- **Support**: team@tensorpool.dev
- **Updates**: Follow us on [Twitter/X](https://x.com/TensorPool)

## Why TensorPool?

- **Simplicity**: Deploy GPU clusters without the complexity of cloud setup, networking, or quotas
- **Flexibility**: Scale from single GPUs to massive multi-node clusters instantly
- **Cost Effective**: Aggregated GPU capacity from multiple providers means better pricing
- **Performance**: High-speed networking and optimized configurations for ML workloads
- **No Lock-in**: Standard SSH access means you can use any tools and frameworks you prefer

Ready to scale your ML training? Get started at [tensorpool.dev](https://tensorpool.dev)!
