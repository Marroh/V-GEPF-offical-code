import numpy as np


def map_coordinates_to_xT(xT, coordinates):
    # Get array dimensions
    rows, cols = xT.shape

    # Initialize a list to store results
    values = []

    # Iterate through coordinate list
    for x, y in coordinates:
        # Map coordinates to array indices
        col_idx = int((x + 1) / 2 * (cols - 1))  # Map x to range 0 to cols-1
        row_idx = int((0.42 - y) / 0.84 * (rows - 1))  # Map y to range 0 to rows-1, note opposite direction

        # Limit indices to array bounds
        col_idx = max(0, min(cols - 1, col_idx))
        row_idx = max(0, min(rows - 1, row_idx))
        # Get corresponding array value
        values.append(xT[row_idx, col_idx])

    return values


if __name__ == '__main__':
    arr = np.load('./xT_maps/WorldCup_xT_5.npy')  # Create a random array with 16 rows and 21 columns
    coordinates = [(0, 0), (-1, -0.42), (1, 0.42), (1, 0)]  # Example coordinates
    result = map_coordinates_to_xT(arr, coordinates)

    print(result)
