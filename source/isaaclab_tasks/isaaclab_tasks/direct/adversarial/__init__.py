# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from isaaclab_tasks.direct.adversarial.sumo.heterogeneous.sumo_stage_2_hetero_by_team_same_critic_no_negative import (
    SumoStage2HeteroByTeamSameCriticNoNegativeEnv,
    SumoStage2HeteroByTeamSameCriticNoNegativeEnvCfg,
)

from . import agents
from .minitank_drone.heterogeneous.minitank_adversarial import MinitankAdversarialEnv, MinitankAdversarialEnvCfg
from .soccer.anymal_c.anymal_c_go_to_point_soccer import AnymalCGoToPointSoccerEnv, AnymalCGoToPointSoccerEnvCfg
from .soccer.anymal_c.anymal_c_soccer_stage_1 import AnymalStage1SoccerEnv, AnymalStage1SoccerEnvCfg
from .soccer.anymal_c.anymal_c_soccer_stage_2 import AnymalStage2SoccerEnv, AnymalStage2SoccerEnvCfg
from .soccer.heterogeneous.anymal_vs_leatherback import AnymalVsLeatherbackSoccerEnv, AnymalVsLeatherbackSoccerEnvCfg
from .soccer.heterogeneous.anymals_vs_leatherbacks import AnymalSoccerHeteroByTeamEnv, AnymalSoccerHeteroByTeamEnvCfg
from .soccer.leatherback.leatherback_soccer_stage_1 import LeatherbackStage1SoccerEnv, LeatherbackStage1SoccerEnvCfg
from .soccer.leatherback.leatherbacks_vs_leatherbacks import (
    LeatherbacksVSLeatherbacksSoccerEnv,
    LeatherbacksVSLeatherbacksSoccerEnvCfg,
)
from .sumo.anymal_c.anymal_c_go_to_point_sumo import AnymalCGoToPointSumo, AnymalCGoToPointSumoCfg
from .sumo.anymal_c.sumo_stage_1_blocks_push import AnymalCBlocksPushEnv, AnymalCBlocksPushEnvCfg
from .sumo.anymal_c.sumo_stage_2_anymal_1v1 import SumoStage2Anymal1v1Env, SumoStage2Anymal1v1EnvCfg
from .sumo.go2_sensor_curriculum import (
    Go2SensorBlocksPushCfg,
    Go2SensorBlocksPushEnv,
    Go2SensorGoToPointCfg,
    Go2SensorGoToPointEnv,
    Go2SensorSumo1v1Cfg,
    Go2SensorSumo1v1Env,
)
from .sumo.heterogeneous.sumo_stage_2_hetero_by_team import SumoStage2HeteroByTeamEnv, SumoStage2HeteroByTeamEnvCfg
from .sumo.heterogeneous.sumo_stage_2_hetero_by_team_same_critic import (
    SumoStage2HeteroByTeamSameCriticEnv,
    SumoStage2HeteroByTeamSameCriticEnvCfg,
)
from .sumo.heterogeneous.sumo_stage_2_hetero_within_team import SumoStage2HeteroEnv, SumoStage2HeteroEnvCfg
from .sumo.leatherback.leatherback_sumo_ma_stage1 import LeatherbackSumoMAStage1Env, LeatherbackSumoMAStage1EnvCfg
from .sumo.quadruped_curriculum import (
    A1BlocksPushCfg,
    A1GoToPointCfg,
    Go1BlocksPushCfg,
    Go1GoToPointCfg,
    Go2BlocksPushCfg,
    Go2GoToPointCfg,
    Go2VsSpotBlocksPushCfg,
    QuadrupedBlocksPushEnv,
    QuadrupedGoToPointSumo,
    SpotBlocksPushCfg,
    SpotGoToPointCfg,
)
from .sumo.quadruped_stage2 import (
    SumoStage2A11v1EnvCfg,
    SumoStage2AnymalVsGo2EnvCfg,
    SumoStage2AnymalVsSpotEnvCfg,
    SumoStage2Go2VsAnymalEnvCfg,
    SumoStage2Go2VsSpotEnvCfg,
    SumoStage2Go11v1EnvCfg,
    SumoStage2Go21v1EnvCfg,
    SumoStage2Quadruped1v1Env,
    SumoStage2Spot1v1EnvCfg,
    SumoStage2SpotVsAnymalEnvCfg,
    SumoStage2SpotVsGo2EnvCfg,
)

##
# Register Gym environments.
##

for stage in ("Locomotion", "Push", "Fight"):
    gym.register(
        id=f"Dog-Sumo-Go2-{stage}-v3",
        entry_point=f"{__name__}.sumo.sumo_v3:Go2Tactics{stage}Env",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.sumo.sumo_v3:Go2Tactics{stage}Cfg"},
    )

for stage, env_name, cfg_name in (
    ("Locomotion", "DogArenaLocomotionEnv", "Go2ArenaLocomotionCfg"),
    ("Push", "DogArenaPushEnv", "Go2ArenaPushCfg"),
    ("Fight", "DogArenaFightEnv", "Go2ArenaFightCfg"),
):
    gym.register(
        id=f"Dog-Sumo-Go2-{stage}-v2",
        entry_point=f"{__name__}.sumo.sumo_v2:{env_name}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sumo.sumo_v2:{cfg_name}",
            "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_sensor_cfg.yaml",
        },
    )

gym.register(
    id="Leatherback-Stage1-Soccer-v0",
    entry_point=LeatherbackStage1SoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackStage1SoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="Unitree-Go2-vs-Spot-Sumo-Stage1-Blocks-Push-v0",
    entry_point=QuadrupedBlocksPushEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2VsSpotBlocksPushCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

for task_id, entry_point, env_cfg in (
    ("Unitree-Go2-Sensor-Go-To-Point-Sumo-v0", Go2SensorGoToPointEnv, Go2SensorGoToPointCfg),
    ("Unitree-Go2-Sensor-Sumo-Stage1-Blocks-Push-v0", Go2SensorBlocksPushEnv, Go2SensorBlocksPushCfg),
    ("Sumo-Stage2-Unitree-Go2-Sensor-1v1-v0", Go2SensorSumo1v1Env, Go2SensorSumo1v1Cfg),
):
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_sensor_cfg.yaml",
        },
    )

gym.register(
    id="Leatherback-Stage2-Soccer-v0",
    entry_point=LeatherbacksVSLeatherbacksSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbacksVSLeatherbacksSoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Hetero_By_Team-v0",
    entry_point=AnymalSoccerHeteroByTeamEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalSoccerHeteroByTeamEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Sumo-Stage2-Hetero-By-Team-v0",
    entry_point=SumoStage2HeteroByTeamEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2HeteroByTeamEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Sumo-Stage2-Hetero-Same-Critic-v0",
    entry_point=SumoStage2HeteroByTeamSameCriticEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2HeteroByTeamSameCriticEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Sumo-Stage2-Hetero-Same-Critic-No-Negative-v0",
    entry_point=SumoStage2HeteroByTeamSameCriticNoNegativeEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2HeteroByTeamSameCriticNoNegativeEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Minitank-Adversarial-Direct-v0",
    entry_point=MinitankAdversarialEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MinitankAdversarialEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Anymal-C-Sumo-Stage1-Blocks-Push-v0",
    entry_point=AnymalCBlocksPushEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCBlocksPushEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Sumo-Stage2-Anymal-1v1-v0",
    entry_point=SumoStage2Anymal1v1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2Anymal1v1EnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

for task_id, env_cfg in (
    ("Sumo-Stage2-Spot-1v1-v0", SumoStage2Spot1v1EnvCfg),
    ("Sumo-Stage2-Unitree-A1-1v1-v0", SumoStage2A11v1EnvCfg),
    ("Sumo-Stage2-Unitree-Go1-1v1-v0", SumoStage2Go11v1EnvCfg),
    ("Sumo-Stage2-Unitree-Go2-1v1-v0", SumoStage2Go21v1EnvCfg),
    ("Sumo-Stage2-Unitree-Go2-vs-Spot-v0", SumoStage2Go2VsSpotEnvCfg),
    ("Sumo-Stage2-Spot-vs-Anymal-v0", SumoStage2SpotVsAnymalEnvCfg),
    ("Sumo-Stage2-Anymal-vs-Unitree-Go2-v0", SumoStage2AnymalVsGo2EnvCfg),
    ("Sumo-Stage2-Spot-vs-Unitree-Go2-v0", SumoStage2SpotVsGo2EnvCfg),
    ("Sumo-Stage2-Anymal-vs-Spot-v0", SumoStage2AnymalVsSpotEnvCfg),
    ("Sumo-Stage2-Unitree-Go2-vs-Anymal-v0", SumoStage2Go2VsAnymalEnvCfg),
):
    gym.register(
        id=task_id,
        entry_point=SumoStage2Quadruped1v1Env,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        },
    )

for model_name, stage0_cfg, stage1_cfg in (
    ("Spot", SpotGoToPointCfg, SpotBlocksPushCfg),
    ("Unitree-A1", A1GoToPointCfg, A1BlocksPushCfg),
    ("Unitree-Go1", Go1GoToPointCfg, Go1BlocksPushCfg),
    ("Unitree-Go2", Go2GoToPointCfg, Go2BlocksPushCfg),
):
    gym.register(
        id=f"{model_name}-Go-To-Point-Sumo-v0",
        entry_point=QuadrupedGoToPointSumo,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": stage0_cfg,
            "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        },
    )
    gym.register(
        id=f"{model_name}-Sumo-Stage1-Blocks-Push-v0",
        entry_point=QuadrupedBlocksPushEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": stage1_cfg,
            "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        },
    )

gym.register(
    id="Anymal-C-Go-To-Point-Sumo",
    entry_point=AnymalCGoToPointSumo,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCGoToPointSumoCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Go_To_Point_Stage_0",
    entry_point=AnymalCGoToPointSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCGoToPointSoccerEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Go_To_Ball_Stage_1",
    entry_point=AnymalStage1SoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalStage1SoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Score_Goals_Stage_2",
    entry_point=AnymalStage2SoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalStage2SoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="leatherback-Sumo-Direct-MA-Stage1-v0",
    entry_point=LeatherbackSumoMAStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoMAStage1EnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC-VS-Leatherback-Soccer-v0",
    entry_point=AnymalVsLeatherbackSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalVsLeatherbackSoccerEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)
