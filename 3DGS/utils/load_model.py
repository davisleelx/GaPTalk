def load_state_dict(model, state_dict, replace=''):
    # load weights (copied from state manager)
    state = model.state_dict()
    to_load = state_dict

    for (k, v) in to_load.items():
        if hasattr(v, 'shape'):
            k = k.replace(replace, '')
            if k not in state:
                print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")

    for (k, v) in state.items():
        k_to_load = replace + k
        if k_to_load not in to_load:
            print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")

        else:
            if state[k].shape == to_load[k_to_load].shape:
                state[k] = to_load[k_to_load]
            else:
                print(f"    - WARNING: Model file contains key {k} ({list(v.shape)}), but to load key {k_to_load} ({list(to_load[k_to_load].shape)})")

    model.load_state_dict(state)
    