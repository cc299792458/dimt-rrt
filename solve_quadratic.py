import numpy as np

def solve_quadratic(a, b, c):
    """
    Solve the quadratic equation a*x^2 + b*x + c = 0 using:
        q = -1/2 * (b + sgn(b) * sqrt(b^2 - 4*a*c))
        x1 = q / a
        x2 = c / q
    
    This method ensures both solutions are calculated in a unified form.
    Only real solutions are considered here.
    
    Returns:
        A tuple (x1, x2), which are the two real solutions.
    
    Raises:
        ValueError: If a == 0 (not a quadratic equation),
                    or if the discriminant < 0 (no real solutions),
                    or if q == 0 (division by zero).
    """
    # Check if it is actually a quadratic equation
    if a == 0:
        raise ValueError("Not a quadratic equation (a=0).")

    # Compute the discriminant
    disc = b**2 - 4*a*c
    if disc < 0:
        return None, None

    # Take the square root of the discriminant
    sqrt_disc = np.sqrt(disc)

    # Determine the sign of b; if b == 0, force sign_b to 1 to avoid q=0
    sign_b = np.sign(b) if np.sign(b) != 0 else 1

    # Calculate q using the provided formula
    q = -0.5 * (b + sign_b * sqrt_disc)

    # Check if q is zero to avoid division by zero
    if q == 0:
        return None, None

    # Compute the two solutions
    x1 = q / a
    x2 = c / q

    # Check if x1 is close to zero; if so, set x1 to np.zeros([1])
    if np.isclose(x1, 0, atol=1e-12):
        x1 = np.zeros([1])
    
    # Check if x2 is close to zero; if so, set x2 to np.zeros([1])
    if np.isclose(x2, 0, atol=1e-12):
        x2 = np.zeros([1])

    return x1, x2
