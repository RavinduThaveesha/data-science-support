def ohe_pd(df: pd.DataFrame, columns: []):
    """
    Onehot encode and convert results into pandas dataframe.

    Args:
      df: Dataframe
      columns: column names which onehot encode requires

    Returns:
      Dataframe with encoded data.
    """

    oh = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
    oh_array = oh.fit_transform(df[columns])
    feature_names = oh.get_feature_names_out()
    df_oh = pd.DataFrame(
            data=oh_array,
            columns=feature_names,
            index=df.index)
    df = pd.concat([df, df_oh], axis=1)
    df.drop(columns, axis=1, inplace=True)
    
    return df