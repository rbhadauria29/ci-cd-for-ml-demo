import argparse
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def get_hp_tuning_results(grid_search: GridSearchCV) -> str:
    """Get the results of hyperparameter tuning in a Markdown table"""
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.sort_values(by="mean_test_score", ascending=False, inplace=True)

    # Extract and split the 'params' column into subcolumns
    params_df = pd.json_normalize(cv_results["params"])

    # Concatenate the params_df with the original DataFrame
    cv_results = pd.concat([cv_results, params_df], axis=1)

    # Get the columns to display in the Markdown table
    markdown_table = cv_results[
        ["rank_test_score", "mean_test_score", "std_test_score"]
        + list(params_df.columns)
    ].to_markdown(index=False)

    return markdown_table


def main():
    # Load the data
    X, y = load_data(PROCESSED_DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=1993)

    # Define the model and hyperparameter search space
    model = RandomForestClassifier()
    param_grid = json.load(open("hp_config.json", "r"))

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Print and save the best hyperparameters
    print("====================Best Hyperparameters==================")
    print(json.dumps(best_params, indent=2))
    print("==========================================================")

    with open("rfc_best_params.json", "w") as outfile:
        json.dump(best_params, outfile)

    # Print and save the results of hyperparameter tuning
    markdown_table = get_hp_tuning_results(grid_search)
    with open("hp_tuning_results.md", "w") as markdown_file:
        markdown_file.write(markdown_table)


if __name__ == "__main__":
    main()
