def check_df(df, top):
    """
    Check if the dataframe is None, capped, or empty

    Args:
        df (DataFrame-like): the dataframe to check
        top (int): the top number of rows to query
    
    Returns:
        {force_break: bool, retry: bool, prev_top: int, new_top: int}
    """
    new_top = top # default new top
    if df is None:
        # if the dataframe is None, therefore the query failed for some reason
        return {
            "force_break": True,
            "retry": False,
            "prev_top": top,
            "new_top": top
        }
    elif len(df) == top:
        # if the dataframe is capped, double the top number of rows to query
        new_top = 2*top
        return {
            "force_break": False,
            "retry": True,
            "prev_top": top,
            "new_top": new_top
        }
    elif len(df) == 0:
        # if the dataframe is empty, it means that there is something wrong.
        # let's force the break and investigate further
        return {
            "force_break": True,
            "retry": False,
            "prev_top": top,
            "new_top": top
        }
    if top > 2*len(df):
        # just for good measure.
        new_top = 2*len(df)
    # no problem, return the new top number of rows to query. Let's go!
    return {
        "force_break": False,
        "retry": False,
        "prev_top": top,
        "new_top": new_top
    }