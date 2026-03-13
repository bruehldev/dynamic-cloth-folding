import os

import cv2
import numpy as np
import pybullet as p


class Camera:
    def __init__(self, image_size, randomization_kwargs):
        # Match call site in cloth_env_pybullet.py
        self.image_size = image_size
        self.randomization_kwargs = randomization_kwargs
        self._cam_cfg = randomization_kwargs["camera_config"]
        self.render_size = tuple(randomization_kwargs["render_size"])  # (W_render, H_render)
        # We run under EGL; render with hardware OpenGL (no fallbacks).
        self._renderer = p.ER_BULLET_HARDWARE_OPENGL
        self.albumentations_transform = None
        if randomization_kwargs.get("albumentations_randomization"):
            import albumentations as A

            cfg = randomization_kwargs["albumentations_config"]
            self.albumentations_transform = A.Compose(
                [
                    A.RGBShift(**cfg["RGBShift"]),
                    A.RandomBrightnessContrast(**cfg["RandomBrightnessContrast"]),
                    A.Blur(**cfg["Blur"]),
                    A.ColorJitter(**cfg["ColorJitter"]),
                    A.GaussianBlur(**cfg["GaussianBlur"]),
                ]
            )

    # -------- episode plumbing (kept short and deterministic) --------
    def begin_episode(self, center_w):
        cfg = self._cam_cfg
        if self.randomization_kwargs["camera_position_randomization"]:
            fmin, fmax = cfg["fovy_range"]
            self._episode_fov = np.random.uniform(fmin, fmax)
        else:
            self._episode_fov = cfg["train_camera_fovy"]
        self.center = np.array(center_w, dtype=float)
        if self.randomization_kwargs["lookat_position_randomization"]:
            r = self.randomization_kwargs["lookat_position_randomization_radius"]
            self.center = self.center + [np.random.uniform(-r, r), np.random.uniform(-r, r), 0.0]
        self._episode_center = self.center
        # Use the configured FOV and the “default” camera type (stable)
        cam_type = self._cam_cfg["type"]
        if cam_type == "all":
            # If any camera-related DR is on, allow random choice. Otherwise, use default.
            is_dr_active = (
                self.randomization_kwargs["camera_position_randomization"]
                or self.randomization_kwargs["lookat_position_randomization"]
            )
            cam_type = np.random.choice(list(cfg["types"].keys())) if is_dr_active else "default"
        # print(f"Camera type for this episode: {cam_type}")
        self.eye, self.up = self._get_eye_from_type(self.center, cam_type)

        # Freeze eye jitter once per episode
        if self.randomization_kwargs["camera_position_randomization"]:
            jx, jy, jz = cfg["jitter_xyz"]
            self.eye = np.array(self.eye) + [
                np.random.uniform(-jx, jx),
                np.random.uniform(-jy, jy),
                np.random.uniform(-jz, jz),
            ]
        self._episode_eye = np.array(self.eye, dtype=float).tolist()
        self._episode_up = self.up

    def _get_eye_from_type(self, center_w, cam_type):
        """Returns eye position and up vector based on camera type."""
        cam_spec = self._cam_cfg["types"][cam_type]
        eye = cam_spec["eye"]
        up = cam_spec["up"]

        # If any camera-related DR is on, treat eye as an offset.
        is_dr_active = (
            self.randomization_kwargs["camera_position_randomization"]
            or self.randomization_kwargs["lookat_position_randomization"]
        )
        if is_dr_active:
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
        near = float(self._cam_cfg["near_clip"])
        far = float(self._cam_cfg["far_clip"])
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        return view_matrix, proj_matrix

    def capture_image(self, center_w):
        # This method uses the "Render Big + Crop" approach (Legacy/Reference)
        assert (
            self.render_size[0] >= self.image_size[0] and self.render_size[1] >= self.image_size[1]
        ), "render_size must be >= image_size; set both in bullet_model_kwargs.py"

        # 1) Render BIG
        W_render, H_render = self.render_size
        view_matrix, proj_matrix = self.get_view_projection_matrices(center_w)

        lights_cfg = self.randomization_kwargs["lights"]
        if self.randomization_kwargs["lights_randomization"]:
            ldir = np.random.uniform(*lights_cfg["direction_range"]).tolist()
            lcol = np.random.uniform(*lights_cfg["color_range"]).tolist()
            shadow = 1
        else:
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
            renderer=self._renderer,
        )
        img = np.reshape(rgba, (H_render, W_render, 4))[:, :, :3].astype("uint8")

        # 2) Center-crop to policy size
        W_out, H_out = self.image_size
        if W_render < W_out or H_render < H_out:
            W_out, H_out = W_render, H_render
        x0 = (W_render - W_out) // 2
        y0 = (H_render - H_out) // 2
        img = img[y0 : y0 + H_out, x0 : x0 + W_out, :]

        # 3) Albumentations
        if self.randomization_kwargs["albumentations_randomization"]:
            img = self.albumentations_transform(image=img)["image"]

        # 4) Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 5) Normalize + flatten
        img = (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return img.flatten().copy()

    def render_rgb_dr(self, center_w, size=None):
        """
        DR renderer used by the policy path.
        - Uses episode-randomized camera (begin_episode)
        - Optional light randomization
        """
        # Render directly at the requested resolution (no crop).
        if size is None:
            W_out, H_out = self.image_size
        else:
            W_out, H_out = size

        view_matrix, proj_matrix = self.get_view_projection_matrices(center_w)

        lights_cfg = self.randomization_kwargs["lights"]
        if self.randomization_kwargs["lights_randomization"]:
            ldir = np.random.uniform(*lights_cfg["direction_range"]).tolist()
            lcol = np.random.uniform(*lights_cfg["color_range"]).tolist()
            shadow = 1
        else:
            ldir = lights_cfg["direction"]
            lcol = lights_cfg["color"]
            shadow = int(lights_cfg["shadows"])

        _, _, rgba, _, _ = p.getCameraImage(
            int(W_out),
            int(H_out),
            view_matrix,
            proj_matrix,
            shadow=shadow,
            lightDirection=ldir,
            lightColor=lcol,
            renderer=self._renderer,
        )
        img = np.reshape(rgba, (int(H_out), int(W_out), 4))[:, :, :3].astype("uint8")
        return img.copy()

    def policy_image(self, center_w):
        # Use DR path for policy-only rendering
        img = self.render_rgb_dr(center_w)

        if self.randomization_kwargs.get("albumentations_randomization"):
            img = self.albumentations_transform(image=img)["image"]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return img.flatten().copy()

    def render_rgb(self, center_w, cam_type="default"):
        W_render, H_render = self.render_size
        # Stable view matches render_size (500x500)
        view_matrix, proj_matrix = self.get_stable_view_projection_matrices(center_w, cam_type)

        lights_cfg = self.randomization_kwargs["lights"]
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
            renderer=self._renderer,
        )
        img = np.reshape(rgba, (H_render, W_render, 4))[:, :, :3].astype("uint8")

        # Center crop for display parity with policy
        W_out, H_out = self.image_size
        x0 = (W_render - W_out) // 2
        y0 = (H_render - H_out) // 2
        return img[y0 : y0 + H_out, x0 : x0 + W_out, :].copy()

    def render_rgb_full(self, center_w, cam_type="default", crop_to_policy=False):
        view_matrix, proj_matrix = self.get_stable_view_projection_matrices(
            center_w, cam_type, crop_to_policy=crop_to_policy
        )
        lights_cfg = self.randomization_kwargs["lights"]
        ldir = lights_cfg["direction"]
        lcol = lights_cfg["color"]
        shadow = int(lights_cfg["shadows"])

        W_render, H_render = self.render_size
        _, _, rgba, _, _ = p.getCameraImage(
            W_render,
            H_render,
            view_matrix,
            proj_matrix,
            shadow=shadow,
            lightDirection=ldir,
            lightColor=lcol,
            renderer=self._renderer,
        )
        img = np.reshape(rgba, (H_render, W_render, 4))[:, :, :3].astype("uint8")
        return img.copy()

    def get_stable_view_projection_matrices(
        self, center_w, cam_type="default", crop_to_policy=False
    ):
        W_render, H_render = self.render_size
        cam_spec = self._cam_cfg["types"][cam_type]

        if cam_spec.get("is_absolute", False):
            eye = cam_spec["eye"]
        else:
            eye = (
                np.array(center_w, dtype=float) + np.array(cam_spec["eye"], dtype=float)
            ).tolist()
        up = cam_spec["up"]

        if "fov" in cam_spec:
            fov = float(cam_spec["fov"])
        else:
            fov = float(self._cam_cfg["train_camera_fovy"])

        if crop_to_policy:
            frac = self.image_size[1] / self.render_size[1]
            fov = float(2.0 * np.degrees(np.arctan(np.tan(np.radians(fov) / 2.0) * frac)))

        aspect = W_render / H_render
        view = p.computeViewMatrix(eye, np.asarray(center_w, dtype=float).tolist(), up)
        proj = p.computeProjectionMatrixFOV(
            fov, aspect, float(self._cam_cfg["near_clip"]), float(self._cam_cfg["far_clip"])
        )
        return view, proj

    def get_stable_camera_setup(self, center_w, cam_type="default", crop_to_policy=False):
        W_render, H_render = self.render_size
        cam_spec = self._cam_cfg["types"][cam_type]
        if cam_spec.get("is_absolute", False):
            eye = cam_spec["eye"]
        else:
            eye = (
                np.array(center_w, dtype=float) + np.array(cam_spec["eye"], dtype=float)
            ).tolist()
        up = cam_spec["up"]
        if "fov" in cam_spec:
            fov = float(cam_spec["fov"])
        else:
            fov = float(self._cam_cfg["train_camera_fovy"])
        if crop_to_policy:
            frac = self.image_size[1] / self.render_size[1]
            fov = float(2.0 * np.degrees(np.arctan(np.tan(np.radians(fov) / 2.0) * frac)))
        aspect = W_render / H_render
        near = float(self._cam_cfg["near_clip"])
        far = float(self._cam_cfg["far_clip"])
        return {"eye": eye, "up": up, "fov": fov, "aspect": aspect, "near": near, "far": far}

    def get_render_crop_params(self):
        W_render, H_render = self.render_size
        return {
            "W_render": int(W_render),
            "H_render": int(H_render),
            "W_out": int(W_render),
            "H_out": int(H_render),
            "x0": 0,
            "y0": 0,
        }

    def set_debug_camera(self, center_w, cam_type="default"):
        eye, _ = self._get_eye_from_type(center_w, cam_type)
        cam_vec = np.array(eye, dtype=np.float32) - np.array(center_w, dtype=np.float32)
        dist = float(np.linalg.norm(cam_vec) or 0.5)
        yaw = float(np.degrees(np.arctan2(cam_vec[1], cam_vec[0])))
        pitch = float(-np.degrees(np.arctan2(cam_vec[2], np.linalg.norm(cam_vec[:2]) + 1e-9)))
        yaw = float(os.getenv("CAM_YAW", yaw))
        pitch = float(os.getenv("CAM_PITCH", pitch))
        dist = float(os.getenv("CAM_DIST", dist))
        p.resetDebugVisualizerCamera(dist, yaw, max(-89.0, min(89.0, pitch)), center_w.tolist())

    def print_gui_camera_as_type(self, name="custom", paste="auto", decimals=5):
        import numpy as np
        import pybullet as p

        info = p.getDebugVisualizerCamera()
        fwd_vec = np.array(info[5], dtype=np.float32)
        dist = float(info[10])
        target = np.array(info[11], dtype=np.float32)
        eye = target - dist * fwd_vec
        center = np.array(getattr(self, "_episode_center", target), dtype=np.float32)
        abs_eye = np.round(eye, decimals).tolist()
        rel_eye = np.round((eye - center), decimals).tolist()
        if paste == "auto":
            is_dr_active = (
                self.randomization_kwargs["camera_position_randomization"]
                or self.randomization_kwargs["lookat_position_randomization"]
            )
            paste = "relative" if is_dr_active else "absolute"
        if paste == "relative":
            print(f'"{name}": {{"eye": {rel_eye}, "up": [0.0, 0.0, 1.0]}},')
        elif paste == "absolute":
            print(f'"{name}": {{"eye": {abs_eye}, "up": [0.0, 0.0, 1.0]}},')
        else:
            print("// absolute:", abs_eye, "   relative:", rel_eye)
