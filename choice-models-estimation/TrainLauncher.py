import LogregModelBlender


def generate_feature_lists(cat, n_lags=3):
    return {
        "outer_feats": (
            [
                "const",
                f"spread_{cat}",
                f"variance_{cat}",
            ]
            + sum([
                [f'log_lag_{lag}_{categ}' for lag in range(n_lags)]
                for categ in (set(["in", "out"]) | set([cat]))
            ], [])
        ),
        "model_names": [
            f'med3_{cat}',
            f'med12_{cat}',
            f'dlz_med12_{cat}',
            f'dlz_geom_avg12_{cat}',
            f'dlz_de02_geom_avg12_{cat}',
        ],
        "model_feats": ["cent", "quant"],
    }


cat = 209

LMB = LogregModelBlender.LogregModelBlender(
    regularization_coef=0.001,
    **generate_feature_lists(cat, n_lags=3)
)
LMB.fit(dataset2, target_col=f"blender_target_{cat}")