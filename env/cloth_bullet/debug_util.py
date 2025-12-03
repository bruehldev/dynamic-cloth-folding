import numpy as np
import pybullet as p


def remove_simple_goal_visual(env):
    """Clear simple EE goal markers."""
    debug_ids = getattr(env, "_simple_goal_debug_ids", None)
    if not debug_ids:
        return
    for _id in debug_ids:
        with np.errstate(all="ignore"):
            p.removeUserDebugItem(_id)
    env._simple_goal_debug_ids = []


def update_simple_goal_visual(env):
    """Draw simple EE goal markers."""
    remove_simple_goal_visual(env)
    if not getattr(env, "simple_ee_task", False):
        return
    goal_I = np.asarray(env.goal, dtype=np.float32)
    goal_W = (env.relative_origin + goal_I).astype(np.float32)
    goal_tip = (goal_W + np.array([0.0, 0.0, 0.08], dtype=np.float32)).tolist()
    color = [1.0, 0.3, 0.1]
    env._simple_goal_debug_ids = []
    env._simple_goal_debug_ids.append(
        p.addUserDebugPoints([goal_W.tolist()], [color], pointSize=12, lifeTime=0)
    )
    env._simple_goal_debug_ids.append(
        p.addUserDebugLine(goal_W.tolist(), goal_tip, color, lineWidth=3.0, lifeTime=0)
    )
    env._simple_goal_debug_ids.append(
        p.addUserDebugText(
            "EE goal",
            goal_tip,
            textColorRGB=color,
            textSize=1.4,
            lifeTime=0,
        )
    )


def spawn_workspace_visual_box(env, origin, limits_min, limits_max, rgba=None):
    """Render a translucent workspace box."""
    if rgba is None:
        rgba = [0, 1, 0, 0.15]
    if getattr(env, "_ws_vis_id", None) is not None:
        try:
            p.removeBody(env._ws_vis_id)
        except Exception:
            pass
        env._ws_vis_id = None

    o = np.array(origin, dtype=float)
    mn = np.array(limits_min, dtype=float)
    mx = np.array(limits_max, dtype=float)
    lo = np.minimum(mn, mx)
    hi = np.maximum(mn, mx)
    half_extents = (hi - lo) * 0.5
    center = o + (lo + hi) * 0.5

    vis = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents.tolist(),
        rgbaColor=rgba,
    )
    env._ws_vis_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=center.tolist(),
        baseOrientation=[0, 0, 0, 1],
    )


def draw_task_visuals(env):
    """Draw constraint origins/targets/goal rays."""
    if env.task is None:
        return
    for _id in getattr(env, "_task_line_ids", []):
        with np.errstate(all="ignore"):
            p.removeUserDebugItem(_id)
    for bid in getattr(env, "_task_marker_ids", []):
        with np.errstate(all="ignore"):
            p.removeBody(bid)
    env._task_line_ids, env._task_marker_ids = [], []

    col_origin = [0.0, 0.2, 1.0]
    col_target = [1.0, 0.0, 0.0]
    col_goal_ray = [0.0, 0.0, 0.0]
    col_goal_pt = [1.0, 1.0, 1.0]
    line_w = 2.0

    sph_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[1, 1, 1, 1])
    sph_target_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.008, rgbaColor=col_target + [1.0]
    )
    sph_goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 1, 1, 1])

    print(env.task.constraints)

    seen_label_sites = set()

    for ci, c in enumerate(env.task.constraints):
        ok = c["origin"]
        tk = c["target"]

        idx_o = env.cloth._site_indices[ok]
        idx_t = env.cloth._site_indices[tk]
        o = np.array(env.cloth.get_position(idx_o), dtype=float)
        t = np.array(env.cloth.get_position(idx_t), dtype=float)
        goal_seg_I = env.goal[ci * 3 : (ci + 1) * 3]
        g = env.relative_origin + goal_seg_I
        same_site = ok == tk

        if not same_site:
            env._task_line_ids.append(
                p.addUserDebugLine(
                    o.tolist(), t.tolist(), [0.9, 0.9, 0.9], lineWidth=line_w, lifeTime=0
                )
            )

        env._task_line_ids.append(
            p.addUserDebugLine(o.tolist(), g.tolist(), col_goal_ray, lineWidth=line_w, lifeTime=0)
        )

        env._task_marker_ids.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=sph_vis,
                basePosition=o.tolist(),
            )
        )
        p.changeVisualShape(env._task_marker_ids[-1], -1, rgbaColor=col_origin + [1.0])
        env._task_marker_ids.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=sph_target_vis,
                basePosition=t.tolist(),
            )
        )
        p.changeVisualShape(env._task_marker_ids[-1], -1, rgbaColor=col_target + [1.0])

        env._task_marker_ids.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=sph_vis,
                basePosition=g.tolist(),
            )
        )
        p.changeVisualShape(env._task_marker_ids[-1], -1, rgbaColor=col_goal_pt + [1.0])

        env._task_marker_ids.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=sph_goal_vis,
                basePosition=g.tolist(),
            )
        )
        p.changeVisualShape(env._task_marker_ids[-1], -1, rgbaColor=col_goal_pt + [1.0])

        if ok not in seen_label_sites:
            env._task_line_ids.append(
                p.addUserDebugText(
                    f"{ok}", o.tolist(), textColorRGB=col_origin, textSize=1.2, lifeTime=0
                )
            )
            seen_label_sites.add(ok)
        target_label_pos = t.tolist()
        if same_site:
            target_label_pos = (t + np.array([0.0, 0.0, 0.02])).tolist()
        if tk not in seen_label_sites:
            env._task_line_ids.append(
                p.addUserDebugText(
                    f"{tk}", target_label_pos, textColorRGB=col_target, textSize=1.2, lifeTime=0
                )
            )
            seen_label_sites.add(tk)
        env._task_line_ids.append(
            p.addUserDebugText(
                f"G{ci}", g.tolist(), textColorRGB=col_goal_pt, textSize=1.2, lifeTime=0
            )
        )
