import pybullet as p
import numpy as np
import os

class Camera:
    """
    Manages camera parameters, view/projection matrices, and image capturing.
    """
    def __init__(self, image_size, fov, randomization_kwargs):
        self.image_size = image_size
        self.fov = fov
        self.randomization_kwargs = randomization_kwargs
        self.albumentations_transform = None

    def _get_eye_from_type(self, center_w, cam_type):
        """
        Calculates the camera's eye position based on its type, with ENV var overrides.
        Returns: tuple (eye_position_list, up_vector_list)
        """
        up = [0.0, 1.0, 0.0]  # Default "up" vector

        if cam_type == "side":
            dx = float(os.getenv("CAM_SIDE_DX", "-0.55"))
            dy = float(os.getenv("CAM_SIDE_DY", "0.00"))
            dz = float(os.getenv("CAM_SIDE_DZ", "0.30"))
        elif cam_type == "front":
            dx = float(os.getenv("CAM_FRONT_DX", "0.00"))
            dy = float(os.getenv("CAM_FRONT_DY", "-0.65"))
            dz = float(os.getenv("CAM_FRONT_DZ", "0.30"))
        elif cam_type == "up":
            dx = float(os.getenv("CAM_UP_DX", "0.00"))
            dy = float(os.getenv("CAM_UP_DY", "0.00"))
            dz = float(os.getenv("CAM_UP_DZ", "0.80"))
        else: # "default"
            up = [0.0, 0.0, 1.0]
            dx, dy, dz = -1.0, -1.0, 1.00

        eye = center_w + np.array([dx, dy, dz], dtype=np.float32)
        return eye.tolist(), up

    def get_view_projection_matrices(self, center_w):
        cfg = self.randomization_kwargs.get("camera_config", {})
        # cam type: "all" -> random pick
        cam_type = cfg.get("type", "default")
        if cam_type == "all":
            cam_type = np.random.choice(["default", "side", "front", "up"])

        eye, up = self._get_eye_from_type(center_w, cam_type)

        # small pose jitter
        jx, jy, jz = cfg.get("jitter_xyz", [0.0, 0.0, 0.0])
        eye = (np.array(eye) + np.array([
            np.random.uniform(-jx, jx),
            np.random.uniform(-jy, jy),
            np.random.uniform(-jz, jz)])).tolist()

        # fovy range
        fovy_range = cfg.get("fovy_range", [self.fov, self.fov])
        self.fov = float(np.random.uniform(*fovy_range))

        aspect = float(self.image_size[0]) / max(1.0, float(self.image_size[1]))
        view_matrix = p.computeViewMatrix(eye, center_w.tolist(), up)
        proj_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, 0.01, 5.0) # Increased far plane
        return view_matrix, proj_matrix

    def capture_image(self, center_w):
        view_matrix, proj_matrix = self.get_view_projection_matrices(center_w)
        W, H = self.image_size

        # Pick renderer based on connection method
        conn = p.getConnectionInfo().get('connectionMethod', p.DIRECT)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if conn == p.GUI else p.ER_TINY_RENDERER

        _, _, rgba, _, _ = p.getCameraImage(W, H, view_matrix, proj_matrix, renderer=renderer)
        img = np.reshape(rgba, (H, W, 4))[:, :, :3].astype("uint8")
        
        # Center crop to the final image_size
        h0 = int(H / 2 - self.image_size[1] / 2)
        w0 = int(W / 2 - self.image_size[0] / 2)
        h0 = max(0, min(h0, H - self.image_size[1]))
        w0 = max(0, min(w0, W - self.image_size[0]))
        img = img[h0:h0 + self.image_size[1], w0:w0 + self.image_size[0], :]
        
        # FIX: Restore the albumentations transform block to match original behavior
        try:
            if hasattr(self, 'albumentations_transform'):
                 img = self.albumentations_transform(image=img)["image"]
        except Exception:
            pass
        
        img = img.astype(np.float32) / 255.0
        img = np.clip(np.nan_to_num(img), 0.0, 1.0)
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