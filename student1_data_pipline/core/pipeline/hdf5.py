from __future__ import annotations

import datetime
import os
from glob import glob
from typing import Tuple

import h5py
import numpy as np
import robosuite as suite


def gather_demonstrations_as_hdf5(directory: str, out_dir: str, env_info: str, hdf5_name: str = "demo.hdf5") -> Tuple[str, int]:
    """
    Convert DataCollectionWrapper raw episode folders into the official robosuite-style HDF5.
    """
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, hdf5_name)

    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")
        num_eps = 0
        env_name = None

        for ep_directory in sorted(os.listdir(directory)):
            ep_path = os.path.join(directory, ep_directory)
            if not os.path.isdir(ep_path):
                continue

            states = []
            actions = []
            success = False

            for state_file in sorted(glob(os.path.join(ep_path, "state_*.npz"))):
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or bool(dic["successful"])

            if len(states) == 0:
                continue

            if success:
                del states[-1]
                assert len(states) == len(actions), f"states/actions mismatch in {ep_path}"

                num_eps += 1
                ep_data_grp = grp.create_group(f"demo_{num_eps}")
                xml_path = os.path.join(ep_path, "model.xml")
                with open(xml_path, "r", encoding="utf-8") as xml_f:
                    xml_str = xml_f.read()

                ep_data_grp.attrs["model_file"] = xml_str
                ep_data_grp.create_dataset("states", data=np.array(states))
                ep_data_grp.create_dataset("actions", data=np.array(actions))

        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name if env_name is not None else "TwoArmLift"
        grp.attrs["env_info"] = env_info
        grp.attrs["total"] = num_eps

    return hdf5_path, num_eps
