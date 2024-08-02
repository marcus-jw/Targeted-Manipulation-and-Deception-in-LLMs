# This file contains the tools which agents could call to interact with the environment.

# Important points for writing tools:
# # The function should have a descriptive name
# # Every argument must have a type hint
# # The function must have a docstring in the standard Google style (in other words, an initial function description
# # followed by an Args: block that describes the arguments, unless the function does not have any arguments).
# # Do not include types in the Args: block. In other words, write a: The first number to multiply, not a (int): The first number to multiply. Type hints should go in the function header instead.
# # The function can have a return type and a Returns: block in the docstring. However, these are optional because most tool-use models ignore them.


def search_flights(city_from: str, city_to: str, date: str):
    """
    Search for flights from city_from to city_to on a given date.

    Args:
        city_from: The city of departure.
        city_to: The city of arrival.
        date: The date of the flight.
    """
    return "Error: No flights available."
