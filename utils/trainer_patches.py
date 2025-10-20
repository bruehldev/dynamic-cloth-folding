def patch_get_diagnostics(trainer, base_trainer):
    """
    Adds 'num train calls' from the inner SACTrainer into ClothSacHERTrainer diagnostics.
    """
    orig_get_diag = getattr(trainer, "get_diagnostics", None)

    def _patched_get_diagnostics():
        d = {}
        if callable(orig_get_diag):
            d = orig_get_diag() or {}
        d["num train calls"] = getattr(base_trainer, "_n_train_steps_total", 0)
        return d

    trainer.get_diagnostics = _patched_get_diagnostics
    return trainer
