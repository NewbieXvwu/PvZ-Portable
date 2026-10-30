"""Build a pack for one label set and read it the way the trainer does (test support only)."""
from pack_training_rows import pack
from pvz_training_pack import archive_fingerprint, pack_options, pack_path
from train_rollout_policy import read_pack


def packed_load(data, labels, wait_keep: float = 1.0, label_weight=None, with_context: bool = False,
                obs: str = "v1", cells: bool = False, aux_horizon: int = 0, packet_head: str = "slot",
                row_observer=None, cache_dir=None, rebuild: bool = False, label_keep=None):
    """Build the pack if this exact input has none yet, then read it the way the trainer does.

    rebuild forces the build even when a pack exists: a changed archive of the same size would
    otherwise keep its old pack, and that is exactly the case a test wants to look at.
    """
    options = pack_options(labels, wait_keep, obs, cells, aux_horizon, packet_head, with_context, label_weight,
                           sources=archive_fingerprint(data, set(labels)), label_keep=label_keep)
    path = pack_path(data, options)
    if rebuild or not path.is_file():
        pack(data, options, cache_dir, 1, None, path, quiet=True, rebuild=rebuild)
    return read_pack(path, data, row_observer)
