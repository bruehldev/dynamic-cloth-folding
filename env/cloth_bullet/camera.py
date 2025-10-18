import numpy as np
import pybullet as p
import math
import cv2
import os


class Camera:
    def __init__(self, image_size, randomization_kwargs):
        self.image_size = image_size            # final policy size, e.g. (100, 100)
        self.fov = 60  # Default field of view
        self.randomization_kwargs = randomization_kwargs
        self.albumentations_transform = None

        # NEW: high-res render size (W_render, H_render)
        self.render_size = tuple(
            self.randomization_kwargs.get("render_size", (320, 240))
        )

        self._episode_center = None
        self._episode_eye = None
        self._episode_fov = None
        if self.randomization_kwargs.get("albumentations_randomization", False):
            import albumentations as A
            self.albumentations_transform = A.Compose(
                [
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                               b_shift_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Blur(blur_limit=7, always_apply=False, p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2,
                                  saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0,
                                   always_apply=False, p=0.5),
                ]
            )

    def begin_episode(self, center_w):
        cfg = self.randomization_kwargs.get("camera_config", {})
        mujoco_fovy = cfg.get("train_camera_fovy", 60)  # fallback to 60 if not set
        fmin, fmax = cfg.get("fovy_range", [mujoco_fovy, mujoco_fovy])
        if self.randomization_kwargs.get("camera_position_randomization", False):
            self._episode_fov = np.random.uniform(fmin, fmax)
        else:
            self._episode_fov = (fmin + fmax) / 2

        # Freeze look-at for the whole episode
        center = np.array(center_w, dtype=float)
        if self.randomization_kwargs.get("lookat_position_randomization", False):
            r = float(self.randomization_kwargs.get("lookat_position_randomization_radius", 0.0))
            center = center + [np.random.uniform(-r, r), np.random.uniform(-r, r), 0.0]
        self._episode_center = center

        # Pick camera type once per episode (support "all")
        cam_type = cfg.get("type", "default")
        if cam_type == "all":
            cam_type = np.random.choice(["default", "side", "front", "up"])
        eye, up = self._get_eye_from_type(center, cam_type)

        # Freeze eye jitter once per episode
        if self.randomization_kwargs.get("camera_position_randomization", False):
            jx, jy, jz = cfg.get("jitter_xyz", [0.0, 0.0, 0.0])
            eye = np.array(eye) + [np.random.uniform(-jx, jx),
                                   np.random.uniform(-jy, jy),
                                   np.random.uniform(-jz, jz)]
        self._episode_eye = np.array(eye, dtype=float).tolist()
        self._episode_up = [0.0, 0.0, 1.0]

    def _get_eye_from_type(self, center_w, cam_type):
        """Returns eye position and up vector based on camera type."""
        # MuJoCo canonical camera positions (from arena.xml)
        if cam_type == "up":
            eye = np.array([0.5, -0.7, 1.1])
        elif cam_type == "front":
            eye = np.array([0.5, -1.0, 0.75])
        elif cam_type == "side":
            eye = np.array([-0.4, -0.7, 0.65])
        else:  # default
            # Use MuJoCo's "eval_camera" or "agentview" as default if you wish, or keep as is
            eye = np.array([-0.45, -1.0, 1.0])
        up = [0.0, 0.0, 1.0]
        return eye.tolist(), up

    def get_view_projection_matrices(self, _center_w_unused):
        # Use frozen episode parameters
        center_w = np.array(self._episode_center, dtype=float)
        eye = np.array(self._episode_eye, dtype=float)
        up = getattr(self, "_episode_up", [0.0, 0.0, 1.0])
        fov = getattr(self, "_episode_fov", self.randomization_kwargs.get("camera_config", {}).get("train_camera_fovy", 60))

        # Match projection to the render buffer to avoid stretching
        aspect = self.render_size[0] / self.render_size[1]
        view_matrix = p.computeViewMatrix(eye.tolist(), center_w.tolist(), up)
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, 0.01, 5.0)
        return view_matrix, proj_matrix

    def capture_image(self, center_w):
        view_matrix, proj_matrix = self.get_view_projection_matrices(center_w)

        # 1) Render BIG
        W_render, H_render = self.render_size
        conn = p.getConnectionInfo().get('connectionMethod', p.DIRECT)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if conn == p.GUI else p.ER_TINY_RENDERER

        # --- MuJoCo-like lighting ---
        lights = self.randomization_kwargs.get("lights", {})
        ldir = lights.get("direction", np.random.uniform(-1, 1, size=3).tolist())
        lcol = lights.get("color", np.random.uniform(0.6, 1.0, size=3).tolist())
        shadow = int(self.randomization_kwargs.get("lights_randomization", False))

        _, _, rgba, _, _ = p.getCameraImage(
            W_render, H_render, view_matrix, proj_matrix,
            shadow=shadow, lightDirection=ldir, lightColor=lcol, renderer=renderer
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
        img = img[y0:y0 + H_out, x0:x0 + W_out, :]

        # 3) Albumentations only if enabled (MuJoCo parity)
        if self.randomization_kwargs.get("albumentations_randomization", False) and self.albumentations_transform is not None:
            img = self.albumentations_transform(image=img)["image"]

        # 4) Grayscale (MuJoCo policy input is gray)
        import cv2
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