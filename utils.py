import numpy as np
import pandas as pd

def k(x, y, h):
  return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-(x - y)**2 / (2 * h**2))


def get_data():
    df = pd.read_csv("military-data-prepared.csv")
    x = df['Women'].to_numpy()
    y = df['Men'].to_numpy()
    return x, y

def remove_duplicates(x, y):
  df = pd.DataFrame({"x": x, "y": y})
  df = df.groupby("x").agg(np.mean).reset_index()
  return df['x'].to_numpy(), df['y'].to_numpy()

def leave_one_out_split(x, y, i):
    x_minus_i = np.concatenate([x[0:i], x[i + 1:]])
    y_minus_i = np.concatenate([y[0:i], y[i + 1:]])
    return x_minus_i, y_minus_i, x[i], y[i]

def get_all_splits(x, y):
    for i in range(len(x)):
        yield leave_one_out_split(x, y, i)

def tune_with_cv(modelClass, modelName, params, x, y):
    # If no params to try, just compute CV score
    if params is None or len(params) == 0:
        error = 0
        for x_minus_i, y_minus_i, xi, yi in get_all_splits(x, y):
            model = modelClass()
            model.fit(x_minus_i, y_minus_i)
            pred = model.predict(np.array([xi]))[0]
            error += (pred - yi)**2
        print(f"Cross-validation error for {modelName}: {error}")
        model = modelClass()
        model.fit(x, y)
        return model, error

    else :
        best_error = float('inf')
        best_param = None
        for p in params:
            error = 0
            for x_minus_i, y_minus_i, xi, yi in get_all_splits(x, y):
                model = modelClass(p)
                model.fit(x_minus_i, y_minus_i)
                pred = model.predict(np.array([xi]))[0]
                error += (pred - yi)**2
            print(f"{modelName} with param {p}: Error = {error}")
            if error < best_error:
                best_error = error
                best_param = p
        print(f"Best param for {modelName} was {best_param}. Fitting model on all data...")
        model = modelClass(best_param)
        model.fit(x, y)
        print("Done.")
        return model, best_error