import numpy as np

def compute_error_for_line_points(bias, deriv, points):
    """
    Return the error of the starting line from the optimum.

    Standard Deviation:
     The error is the average of the sum of the squares of the 
     distances for all points.
       from i to N: Error(m,b) = 1/N * SUM( SQR(yi-(m*xi + b)) )
    """
    return (
        sum(pow(p[1] - (deriv * p[0] + bias), 2) for p in points)
        ) / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    """
    Run the optimization with gradient descent
    """
    b = starting_b
    m = starting_m

    def step_gradient(current_b, current_m, points, learning_r):
        """
        Math for gradient descent.

        Partial derivative of b:
         delta over delta(b) = -(2/N) * SUM(yi - (m*xi + b))

        Partial derivative of m:
         delta over delta(m) = -(2/N) * SUM(xi * (yi-(m*xi + b)))
        """
        b_gradient = 0
        m_gradient = 0

        N = len(points)

        for p in points:
            x, y = p[0], p[1]
            # direction with respect to b and m
            # computing the partial derivatives for b and m (component of the direction
            # to go to find the local minimum) of the error function
            b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
            m_gradient += -(2/N) * (x * (y - ((current_m * x) + current_b)))

        # update b and m using partial derivatives

        new_b = current_b - (learning_r * b_gradient)
        new_m = current_m - (learning_r * m_gradient)
        return new_b, new_m

    for i in range(num_iterations):
        # update b and m with more accurate values

        b, m = step_gradient(b, m, np.array(points), learning_rate)

    return b, m

def run():
    # 1. get points from the file
    points = np.genfromtxt('data.csv', delimiter=',')

    #
    # 2. define hyper-parameters
    #
    learning_rate = 0.0001  # how fast should the model converge
    # y = normal * x + bias
    initial_bias = 0
    initial_normal = 0
    num_iterations = 1000  # depends on the size of the dataset, in this case is only 100 values

    #
    # 3. Train (fit) the model
    #
    print('starting gradient descent at b={}, m={}, error={}'.format(
            initial_bias, initial_normal, 
            compute_error_for_line_points(initial_bias, initial_normal, points)
        )
    )

    b, m = gradient_descent_runner(
            points, initial_bias, initial_normal,
            learning_rate, num_iterations
        )

    print('gradient descent result after {} iteration: ' 
          'at b={}, m={}, error={}'.format(
            num_iterations, b, m, 
            compute_error_for_line_points(b, m, points)
        )
    )



if __name__ == '__main__':
    run()