import os

import cv2
import numpy as np
import pybullet as p


class Camera:
    def __init__(self, image_size, randomization_kwargs):
        self.image_size = image_size  # final policy size, e.g. (100, 100)
        self.randomization_kwargs = randomization_kwargs
        self.albumentations_transform = None
        self.enable_dr = self.randomization_kwargs["enable_dr"]
        # Cache camera config block for convenience
        self._cam_cfg = self.randomization_kwargs["camera_config"]

        # NEW: high-res render size (W_render, H_render)
        self.render_size = tuple(self.randomization_kwargs["render_size"])

        self._episode_center = None
        self._episode_eye = None
        self._episode_fov = None
        # Only construct augmentation pipeline if DR + flag are ON
        if self.enable_dr and self.randomization_kwargs["albumentations_randomization"]:
            import albumentations as A

            cfg = self.randomization_kwargs["albumentations_config"]
            self.albumentations_transform = A.Compose(
                [
                    A.RGBShift(**cfg["RGBShift"]),
                    A.RandomBrightnessContrast(**cfg["RandomBrightnessContrast"]),
                    A.Blur(**cfg["Blur"]),
                    A.ColorJitter(**cfg["ColorJitter"]),
                    A.GaussianBlur(**cfg["GaussianBlur"]),
                ]
            )

    def begin_episode(self, center_w):
        cfg = self._cam_cfg
        if self.enable_dr and self.randomization_kwargs["camera_position_randomization"]:
            fmin, fmax = cfg["fovy_range"]
            self._episode_fov = np.random.uniform(fmin, fmax)
        else:
            self._episode_fov = cfg["train_camera_fovy"]

        # Freeze look-at for the whole episode
        center = np.array(center_w, dtype=float)
        if self.enable_dr and self.randomization_kwargs["lookat_position_randomization"]:
            r = self.randomization_kwargs["lookat_position_randomization_radius"]
            center = center + [np.random.uniform(-r, r), np.random.uniform(-r, r), 0.0]
        self._episode_center = center

        # Pick camera type once per episode (support "all")
        cam_type = cfg["type"]
        if cam_type == "all":
            cam_type = np.random.choice(list(cfg["types"].keys())) if self.enable_dr else "default"
        # print(f"Camera type for this episode: {cam_type}")
        eye, up = self._get_eye_from_type(center, cam_type)

        # Freeze eye jitter once per episode
        if self.enable_dr and self.randomization_kwargs["camera_position_randomization"]:
            jx, jy, jz = cfg["jitter_xyz"]
            eye = np.array(eye) + [
                np.random.uniform(-jx, jx),
                np.random.uniform(-jy, jy),
                np.random.uniform(-jz, jz),
            ]
        self._episode_eye = np.array(eye, dtype=float).tolist()
        self._episode_up = up

    def _get_eye_from_type(self, center_w, cam_type):
        """Returns eye position and up vector based on camera type."""
        cam_spec = self._cam_cfg["types"][cam_type]
        eye = cam_spec["eye"]
        up = cam_spec["up"]

        if self.enable_dr:
            # When DR is ON, treat eye as an offset from the cloth center.
            return (np.array(center_w) + np.array(eye)).tolist(), up
        else:
            # When DR is OFF, use the eye position as a fixed world coordinate for consistency
            return eye, up

    def get_view_projection_matrices(self, _center_w_unused):
        # Use frozen episode parameters
        center_w = np.array(self._episode_center, dtype=float)
        eye = np.array(self._episode_eye, dtype=float)
        up = self._episode_up
        fov = self._episode_fov

        # Match projection to the render buffer to avoid stretching
        aspect = self.render_size[0] / self.render_size[1]
        view_matrix = p.computeViewMatrix(eye.tolist(), center_w.tolist(), up)
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, 0.01, 5.0)
        return view_matrix, proj_matrix

    def capture_image(self, center_w):
        view_matrix, proj_matrix = self.get_view_projection_matrices(center_w)

        # 1) Render BIG
        W_render, H_render = self.render_size
        conn = p.getConnectionInfo().get("connectionMethod", p.DIRECT)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if conn == p.GUI else p.ER_TINY_RENDERER

        # --- MuJoCo-like lighting ---
        lights_cfg = self.randomization_kwargs["lights"]
        if self.enable_dr and self.randomization_kwargs["lights_randomization"]:
            # Randomized lights
            ldir = np.random.uniform(*lights_cfg["direction_range"]).tolist()
            lcol = np.random.uniform(*lights_cfg["color_range"]).tolist()
            shadow = 1
        else:
            # Deterministic/stable lights
            ldir = lights_cfg["direction"]
            lcol = lights_cfg["color"]
            shadow = int(lights_cfg["shadows"])

        _, _, rgba, _, _ = p.getCameraImage(
            W_render,
            H_render,
            view_matrix,
            proj_matrix,
            shadow=shadow,
            lightDirection=ldir,
            lightColor=lcol,
            renderer=renderer,
        )
        img = np.reshape(rgba, (H_render, W_render, 4))[:, :, :3].astype("uint8")

        # 2) Center-crop to policy size
        W_out, H_out = self.image_size  # final size, e.g. (100, 100)
        # (assert bigger-or-equal so crop is valid)
        if W_render < W_out or H_render < H_out:
            # fall back: if someone set a tiny render_size by mistake, just skip crop
            W_out, H_out = W_render, H_render
        x0 = (W_render - W_out) // 2
        y0 = (H_render - H_out) // 2
        img = img[y0 : y0 + H_out, x0 : x0 + W_out, :]

        # 3) Albumentations only if enabled (MuJoCo parity)
        if self.enable_dr and self.randomization_kwargs["albumentations_randomization"]:
            img = self.albumentations_transform(image=img)["image"]

        # 4) Grayscale (MuJoCo policy input is gray)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 5) Normalize + flatten
        img = (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return img.flatten().copy()

    def set_debug_camera(self, center_w, cam_type="default"):
        """Sets the GUI debug camera to look at a target, with ENV var overrides."""
        # Use the provided cam_type, which will be hardcoded from the env.
        eye, _ = self._get_eye_from_type(center_w, cam_type)
        cam_vec = np.array(eye, dtype=np.float32) - np.array(center_w, dtype=np.float32)
        dist = float(np.linalg.norm(cam_vec) or 0.5)
        yaw = float(np.degrees(np.arctan2(cam_vec[1], cam_vec[0])))
        pitch = float(-np.degrees(np.arctan2(cam_vec[2], np.linalg.norm(cam_vec[:2]) + 1e-9)))

        # Allow manual tweaks via environment variables
        yaw = float(os.getenv("CAM_YAW", yaw))
        pitch = float(os.getenv("CAM_PITCH", pitch))
        dist = float(os.getenv("CAM_DIST", dist))

        p.resetDebugVisualizerCamera(dist, yaw, max(-89.0, min(89.0, pitch)), center_w.tolist())

    def print_gui_camera_as_type(self, name="custom", paste="auto", decimals=5):
        """
        Print a '"name": {"eye": [...], "up": [0,0,1]},' snippet based on the current GUI camera.

        paste:
        - "relative": store an OFFSET from the episode center (use this when DR is ON)
        - "absolute": store an absolute world position (use when DR is OFF / fixed cam)
        - "auto"    : picks relative if self.enable_dr else absolute
        """
        import numpy as np
        import pybullet as p

        # PyBullet: [w,h,view,proj,upVec,forwardVec,hor,vert,yaw,pitch,dist,target]
        info = p.getDebugVisualizerCamera()
        up_vec = np.array(info[4], dtype=np.float32)  # noqa: F841
        fwd_vec = np.array(info[5], dtype=np.float32)  # points from CAMERA -> TARGET
        dist = float(info[10])
        target = np.array(info[11], dtype=np.float32)

        # This relation is exact and avoids yaw/pitch sign gotchas:
        eye = target - dist * fwd_vec

        # Absolute vs relative (offset)
        center = np.array(getattr(self, "_episode_center", target), dtype=np.float32)
        abs_eye = np.round(eye, decimals).tolist()
        rel_eye = np.round((eye - center), decimals).tolist()

        if paste == "auto":
            paste = "relative" if getattr(self, "enable_dr", False) else "absolute"

        if paste == "relative":
            print(f'"{name}": {{"eye": {rel_eye}, "up": [0.0, 0.0, 1.0]}},')
        elif paste == "absolute":
            print(f'"{name}": {{"eye": {abs_eye}, "up": [0.0, 0.0, 1.0]}},')
        else:
            print("// absolute:", abs_eye, "   relative:", rel_eye)
