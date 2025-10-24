from rlkit.samplers.eval_suite import eval_suite, real_corner_prediction_test, success_rate_test


def make_eval_suite(randomized_env, eval_policy, env_keys, variant):
    success_test = success_rate_test.SuccessRateTest(
        env=randomized_env,
        policy=eval_policy,
        keys=env_keys,
        name="randomized_cloth",
        metric_keys=[
            "success_rate",
            "corner_distance",
            "corner_0",
            "corner_1",
            "corner_2",
            "corner_3",
            "corner_sum_error",
        ],
        **variant["eval_kwargs"],
    )
    real_corner_test = real_corner_prediction_test.RealCornerPredictionTest(
        env=randomized_env,
        policy=eval_policy,
        keys=env_keys,
        name="real_corner_error",
        metric_keys=["corner_error"],
        **variant["eval_kwargs"],
    )
    return eval_suite.EvalTestSuite(tests=[success_test, real_corner_test])
