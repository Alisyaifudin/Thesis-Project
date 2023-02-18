def append_name(element, name):
    """
    Append a table name to a column name.
    Args:
        element (str): The column name.
        name (str): The table name.
    Returns:
        str: The column name with the table name appended.
    Example usage:
        [in]:  append_name("id", "users")

        [out]: users.\"id\"

        [in]:  append_name("id AS user_id", "users")

        [out]: users.\"id\" AS user_id
    """
    string = element.split(" AS ")
    if(len(string) == 1):
        return f"{name}.\"{element}\""
    else:
        return f"{name}.\"{string[0]}\" AS {string[1]}"