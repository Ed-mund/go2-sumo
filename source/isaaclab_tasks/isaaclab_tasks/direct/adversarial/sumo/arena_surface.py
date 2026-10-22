"""Deterministic, symmetric collision meshes with procedural surface patterns.

No generated bitmap assets: color is authored per triangle, and the same mesh is
used for rendering and contact. Variants are fixed per environment during a run.
"""

import numpy as np


def surface_mesh(seed, index, stage):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    axis = np.linspace(-5, 5, 81, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    amplitude = 0.0 if index % 4 == 0 else rng.uniform(0.008, 0.04 if stage == 0 else 0.025)
    wave = (1 - np.cos(x * rng.uniform(2, 4)) * np.cos(y * rng.uniform(2, 4))) / 2
    # Flat central start pad; terrain and color are invariant under 180-degree rotation.
    envelope = np.clip((np.hypot(x, y) - 0.6) / 0.5, 0, 1)
    z = (0.001 + amplitude * wave * envelope).astype(np.float32)
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    cell = np.arange(80)[:, None] * 81 + np.arange(80)[None, :]
    triangles = np.stack((cell, cell + 81, cell + 82, cell, cell + 82, cell + 1), axis=-1).reshape(-1, 3)
    centres = vertices[triangles].mean(1)
    palette = rng.uniform(0.12, 0.65, 3)
    pattern = index % 3
    if pattern == 0:
        detail = np.cos(centres[:, 0] * 8) * np.cos(centres[:, 1] * 8)
    elif pattern == 1:
        detail = np.cos((centres[:, 0] + centres[:, 1]) * 12)
    else:
        detail = np.cos(centres[:, 0] * 17) * np.cos(centres[:, 1] * 19)
    colors = np.clip(palette[None, :] * (1 + 0.30 * detail[:, None]), 0, 1).astype(np.float32)
    radius = 4.5 if stage == 0 else 3.0
    band = np.abs(np.linalg.norm(centres[:, :2], axis=1) - radius) < 0.10
    colors[band] = 0.94  # always-visible rule boundary, never a physical curb
    return vertices, triangles, colors, float(z.max())


def install_surfaces(env):
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
    import isaaclab.sim as sim_utils

    stage = omni.usd.get_context().get_stage()
    env._surface_max = []
    for index in range(env.num_envs):
        variant = index + env.cfg.arena_variant_offset
        vertices, triangles, colors, maximum = surface_mesh(env.cfg.seed, variant, env.cfg.sumo_stage)
        path = f"/World/envs/env_{index}/ArenaSurface"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(triangles.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorPrimvar("uniform").Set(Vt.Vec3fArray.FromNumpy(colors))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        material = UsdShade.Material.Define(stage, path + "/Material")
        shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        reader = UsdShade.Shader.Define(stage, path + "/Material/Color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        friction = 0.65 + (variant % 5) * 0.07
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(friction + 0.1)
        physics.CreateDynamicFrictionAttr(friction)
        physics.CreateRestitutionAttr(0.0)
        sim_utils.bind_physics_material(path, str(material.GetPath()))
        env._surface_max.append(maximum)
    # Run/seed variation is shared by both opponents, avoiding side-specific lighting cues.
    light = stage.GetPrimAtPath("/World/Light")
    rng = np.random.default_rng(env.cfg.seed + 713)
    light.GetAttribute("inputs:intensity").Set(float(rng.uniform(1200, 2500)))
    light.GetAttribute("inputs:color").Set(Gf.Vec3f(*rng.uniform(0.75, 1.0, 3).tolist()))
