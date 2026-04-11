import os
import cloudpickle as pickle
import absl.app
import absl.flags as flags

from diffusion.trainer import DiffusionTrainer, SamplerPolicy

FLAGS = flags.FLAGS
flags.DEFINE_string("warmstart_ckpt", "", "Path to offline model_final.pkl or model_*.pkl")

def main(argv):
    if not FLAGS.warmstart_ckpt:
        raise ValueError("--warmstart_ckpt is required")
    if not os.path.exists(FLAGS.warmstart_ckpt):
        raise FileNotFoundError(FLAGS.warmstart_ckpt)

    trainer = DiffusionTrainer()
    trainer._setup()  # build dataset/env/logger once

    with open(FLAGS.warmstart_ckpt, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "agent" in obj:
        trainer._agent = obj["agent"]
    else:
        trainer._agent = obj

    trainer._sampler_policy = SamplerPolicy(trainer._agent.policy, trainer._agent.qf)

    # prevent train() from calling _setup() again and overwriting loaded agent
    trainer._setup = lambda: None

    trainer.train()

if __name__ == "__main__":
    absl.app.run(main)
