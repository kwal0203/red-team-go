import sqlite3


def get_random_samples(db_path, num_samples_per_dataset=10) -> list:
    """
    Retrieves random samples from the database, with an equal number from each dataset.

    Args:
      db_path: Path to the SQLite database.
      num_samples_per_dataset: Number of samples to retrieve from each dataset.

    Returns:
      A list of tuples, where each tuple contains (dataset_name, text).
      Returns an empty list if an error occurs or no data is found.
    """
    try:
        print(f"db_path: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        datasets = [
            "toxic_chat",
            "advprompt",
            "real_toxicity_prompts",
            "decoding_trust",
            "fft",
            "cot_bias",
        ]
        samples = []

        for dataset in datasets:
            cursor.execute(
                "SELECT dataset, prompt FROM prompts WHERE dataset = ? ORDER BY RANDOM() LIMIT ?",
                (dataset, num_samples_per_dataset),
            )
            dataset_samples = cursor.fetchall()
            result = [dict(row) for row in dataset_samples]
            samples.extend(result)

        conn.close()
        return samples

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []


def get_samples(db_path: str, num_samples: int) -> list:
    """
    Retrieves rows from the database starting from the first row, up to num_samples rows.

    Args:
      db_path: Path to the SQLite database.
      num_samples: Number of rows to retrieve.

    Returns:
      A list of tuples containing the retrieved rows.
      Returns an empty list if an error occurs or no data is found.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch the requested number of rows or all rows if fewer are available
        cursor.execute("SELECT dataset, prompt FROM prompts LIMIT ?", (num_samples,))
        samples = cursor.fetchall()

        conn.close()
        return samples

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []
