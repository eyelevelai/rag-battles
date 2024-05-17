def move_columns(df):
    column_to_move = df.pop("query")
    df.insert(0, "query", column_to_move)

    column_to_move = df.pop("expected_response")
    df.insert(1, "expected_response", column_to_move)

    column_to_move = df.pop("response")
    df.insert(2, "response", column_to_move)

    column_to_move = df.pop("approach")
    df.insert(3, "approach", column_to_move)

    column_to_move = df.pop("partition_id")
    df.insert(4, "partition_id", column_to_move)

    column_to_move = df.pop("partition_name")
    df.insert(5, "partition_name", column_to_move)

    column_to_move = df.pop("problem_type")
    df.insert(6, "problem_type", column_to_move)

    column_to_move = df.pop("context_modality")
    df.insert(7, "context_modality", column_to_move)

    column_to_move = df.pop("retrieval_count")
    df.insert(8, "retrieval_count", column_to_move)

    column_to_move = df.pop("source")
    df.insert(8, "source", column_to_move)

    df.pop("context_file")

    check = ["query", "expected_response", "response", "approach", "partition_id", "partition_name", "problem_type", "context_modality", "retrieval_count", "source"]
    if set(df.columns) != set(check):
        print(df.columns)
        raise ValueError('incompatible columns')

    return df