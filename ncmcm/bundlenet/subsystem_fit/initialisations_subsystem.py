"""
@authors:
Akshey Kumar
Vittorio Boarini
"""
import copy
import uuid
import torch

def best_of_5_runs(x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data, device):
    """
    Initialises BunDLe net with the best of 5 runs

    Performs 100 epochs of training for 5 random model initialisations
    and picks the model with the lowest loss
    """
    if validation_data is None:
        import warnings

        warnings.warn(
            "No validation data given. Will proceed to use train dataset loss as deciding factor for the best model"
        )
        validation_data = (x_train, b_train_1)

    best_loss = float('inf')
    best_weights = None

    for i in range(5):
        from .bundlenet_subsystem import train_model
        model_ = copy.deepcopy(model)

        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model_,
            b_type=b_type,
            gamma=gamma,
            learning_rate=learning_rate,
            n_epochs=100,
            validation_data=validation_data,
            initialisation=None,
            device=device,
            report_ray_tune=False,
        )

        # Store the best weights in memory
        current_loss = test_history[-1, -1]
        print("model:", i, "val loss:", current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = model_.state_dict()

    # Set the best weights back to the original model
    model.load_state_dict(best_weights)
    return model


def best_of_n_runs(n, n_epochs, x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data, device):
    """
    Initialises BunDLe net with the best of n runs

    Performs n_epochs epochs of training for n random model initialisations
    and picks the model with the lowest loss
    """
    if validation_data is None:
        import warnings

        warnings.warn(
            "No validation data given. Will proceed to use train dataset loss as deciding factor for the best model"
        )
        validation_data = (x_train, b_train_1)

    best_loss = float('inf')
    best_weights = None

    for i in range(n):
        from .bundlenet_subsystem import train_model
        model_ = copy.deepcopy(model)
        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model_,
            b_type=b_type,
            gamma=gamma,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            validation_data=validation_data,
            initialisation=None,
            device=device,
            report_ray_tune=False,
        )

        # Store the best weights in memory
        current_loss = test_history[-1, -1]
        print("model:", i, "val loss:", current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = model_.state_dict()

    # Set the best weights back to the original model
    model.load_state_dict(best_weights)
    return model
