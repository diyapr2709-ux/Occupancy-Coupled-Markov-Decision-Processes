"""Gymnasium environments. Importing this package registers the ids below.

    import ocmdp.envs
    env = gymnasium.make("ocmdp/TwoDrug-v0")
"""

from gymnasium.envs.registration import register

from ocmdp.envs.toy import ToyConfig, ToyOCMDPEnv

register(id="ocmdp/TwoDrug-v0", entry_point="ocmdp.envs.toy:ToyOCMDPEnv", kwargs={"config": ToyConfig.two_drug()})
register(id="ocmdp/PairedDrugs-v0", entry_point="ocmdp.envs.toy:ToyOCMDPEnv", kwargs={"config": ToyConfig.paired()})

__all__ = ["ToyConfig", "ToyOCMDPEnv"]
