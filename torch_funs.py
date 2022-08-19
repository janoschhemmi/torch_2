##

# export
def prepare_idxs(o, shape=None):
    if o is None:
        return slice(None)
    elif is_slice(o) or isinstance(o, Integral):
        return o
    else:
        if shape is not None:
            return np.array(o).reshape(shape)
        else:
            return np.array(o)


def prepare_sel_vars_and_steps(sel_vars=None, sel_steps=None, idxs=False):
    sel_vars = prepare_idxs(sel_vars)
    sel_steps = prepare_idxs(sel_steps)
    if not is_slice(sel_vars) and not isinstance(sel_vars, Integral):
        if is_slice(sel_steps) or isinstance(sel_steps, Integral):
            _sel_vars = [sel_vars, sel_vars.reshape(1, -1)]
        else:
            _sel_vars = [sel_vars.reshape(-1, 1), sel_vars.reshape(1, -1, 1)]
    else:
        _sel_vars = [sel_vars] * 2
    if not is_slice(sel_steps) and not isinstance(sel_steps, Integral):
        if is_slice(sel_vars) or isinstance(sel_vars, Integral):
            _sel_steps = [sel_steps, sel_steps.reshape(1, -1)]
        else:
            _sel_steps = [sel_steps.reshape(1, -1), sel_steps.reshape(1, 1, -1)]
    else:
        _sel_steps = [sel_steps] * 2
    if idxs:
        n_dim = np.sum([isinstance(o, np.ndarray) for o in [sel_vars, sel_steps]])
        idx_shape = (-1,) + (1,) * n_dim
        return _sel_vars, _sel_steps, idx_shape
    else:
        return _sel_vars[0], _sel_steps[0]


def apply_sliding_window(
        data,  # and array-like object with the input data
        window_len: int | list,  # sliding window length. When using a list, use negative numbers and 0.
        horizon: int | list = 0,  # horizon
        x_vars: int | list | None = None,  # indices of the independent variables
        y_vars: int | list | None = None,
        # indices of the dependent variables (target). [] means no y will be created. None means all variables.
):
    "Applies a sliding window on an array-like input to generate a 3d X (and optionally y)"

    ## test
    data = subset
    window_len = 21
    x_vars = ["1","2","3","4","5"]
    horizon = 0

    if isinstance(data, pd.DataFrame): data = data.to_numpy()
    if isinstance(window_len, list):
        assert np.max(window_len) == 0
        x_steps = abs(np.min(window_len)) + np.array(window_len)
        window_len = abs(np.min(window_len)) + 1
    else:
        x_steps = None

    # 5 layer of ts shape = 5 / n /21
    X_data_windowed = np.lib.stride_tricks.sliding_window_view(data, window_len, axis=0)
    X_data_windowed = np.lib.stride_tricks.sliding_window_view(data, (len(x_vars),window_len ))

    # X
    sel_x_vars, sel_x_steps = prepare_sel_vars_and_steps(x_vars, x_steps)
    if horizon == 0:
        X = X_data_windowed[:, sel_x_vars, sel_x_steps]
    else:
        X = X_data_windowed[:-np.max(horizon):, sel_x_vars, sel_x_steps]
    if x_vars is not None and isinstance(x_vars, Integral):
        X = X[:, None]  # keep 3 dim

    # y
    if y_vars == []:
        y = None
    else:
        if isinstance(horizon, Integral) and horizon == 0:
            y = data[-len(X):, y_vars]
        else:
            y_data_windowed = np.lib.stride_tricks.sliding_window_view(data, np.max(horizon) + 1, axis=0)[-len(X):]
            y_vars, y_steps = prepare_sel_vars_and_steps(y_vars, horizon)
            y = np.squeeze(y_data_windowed[:, y_vars, y_steps])
    return X, y