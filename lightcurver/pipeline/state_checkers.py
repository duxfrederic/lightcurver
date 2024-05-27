# here we'll define simple functions that check the health of our products.
# can be beneficial to stop the pipeline if an early stage fails.

from ..structure.database import get_count_based_on_conditions
from ..structure.user_config import get_user_config


def check_plate_solving():
    user_config = get_user_config()
    plate_solved = get_count_based_on_conditions(conditions='plate_solved = 1 and eliminated = 0', table='frames')
    total = get_count_based_on_conditions(conditions='eliminated = 0', table='frames')
    min_success_fraction = user_config['plate_solving_min_success_fraction']
    reasonable_loss = plate_solved / total >= min_success_fraction
    if not reasonable_loss:
        message = "The plate solving failed too often given your config's plate_solving_min_success_fraction"
        message += "There might be a problem, or there might just be a lot of difficult images"
        message += "Please investigate"
    else:
        message = "Plate solving succeeded often enough."
    return reasonable_loss, message


